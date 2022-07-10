import math

from torch import nn
import torch
import torch.nn.functional as F


from model.sub_modules.video_encoder import VideoInitializer, TCN
from model.sub_modules.audio_encoder import load_DAVEnet1, load_DAVEnet2, AudioInitializer
from modules.attention_layers import TanhAttention
from .self_attention import EncoderLayer, Encoder1
from model.CPC_model import CPC
import threading
from utils.helper import sequence_mask
import numpy as np
class RNNEncoder(nn.Module):
    def __init__(self, video_dim, d_model, num_layers):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model

        self.visual_rnn = nn.LSTM(video_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, visual_feature):

        video_output, _ = self.visual_rnn(visual_feature)
        return  video_output

class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)

class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4)
        self.encoder = Encoder1(self.encoder_layer, num_layers=2,hidden_size=d_model)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature



class MomentRetrievalModel(nn.Module):
    def __init__(self, model_dim, chunk_num, use_negative, text_config, video_config, **kwargs):
        super().__init__()
        self.chunk_num = chunk_num
        self.model_dim = model_dim

        self.use_negative = use_negative
        
        self.video_encoder = VideoInitializer(**video_config)
        self.audio_encoder1 = load_DAVEnet1()
        self.audio_encoder2 = load_DAVEnet2()
        self.audio_rnn = AudioInitializer(input_dim=256, hidden_dim=256)
        #self.video_encoder = TCN(**video_config)


        self.core_dim = model_dim // chunk_num
        self.video_audio_attn = TanhAttention(self.core_dim)
        self.feed_forward = nn.Linear(4 * self.core_dim, self.core_dim)


        self.video_self_att = InternalTemporalRelationModule(input_dim=self.core_dim, d_model=self.core_dim)
        self.audio_self_att = InternalTemporalRelationModule(input_dim=self.core_dim, d_model=self.core_dim)
        self.d_model = (self.core_dim)/2
        self.start_rnn = RNNEncoder(video_dim=self.core_dim, d_model=self.core_dim//2, num_layers=1)
        self.end_rnn = RNNEncoder(video_dim=self.core_dim, d_model=self.core_dim//2, num_layers=1)
        self.inside_rnn = RNNEncoder(video_dim=self.core_dim, d_model=self.core_dim//2, num_layers=1)

        self.start_block = nn.Sequential(
            Conv1D(in_dim=2 * self.core_dim, out_dim=self.core_dim, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.Dropout(0.2),
            nn.ReLU(),
            Conv1D(in_dim=self.core_dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)

        )
        self.end_block = nn.Sequential(
            Conv1D(in_dim=2 * self.core_dim, out_dim=self.core_dim, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.Dropout(0.2),
            nn.ReLU(),
            Conv1D(in_dim=self.core_dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.inside_block = nn.Sequential(
            Conv1D(in_dim=2 * self.core_dim, out_dim=self.core_dim, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.Dropout(0.2),
            nn.ReLU(),
            Conv1D(in_dim=self.core_dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )


        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)

        self.audio_conv1 = nn.Sequential(
            nn.Conv1d(128, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 256, 4, 2, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )

        self.audio_conv2 = nn.Sequential(
            nn.Conv1d(256, 512, 4, 2, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 1024, 4, 2, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )

        self.audio_feed_forward = nn.Sequential(
            nn.LayerNorm(256),
            nn.ReLU(True),
            nn.Linear(256, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(True),
            nn.Linear(256, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(True),
            nn.Linear(256, 256),
        )

        self.audio_new_rnn = nn.GRU(256, 256, batch_first=True)

        # self.CPC_model = CPC(in_channels=128, channels=256, n_embeddings=512, z_dim=256, c_dim=256, n_prediction_steps=3, video_config=video_config)
        # state_dict = torch.load('./checkpoints/clean_CPC/model-188.pt')
        # parameters = state_dict['model_parameters']
        # self.CPC_model.load_state_dict(parameters)





    def fuse_and_route(self, visual_feat, textual_feat, audio_feat, textual_mask, audio_mask, audio_length,start, end, epoch):
        batch_size, visual_len, _ = visual_feat.size()
        orginal_visual_feat = visual_feat
        visual_feat = self.video_encoder(visual_feat)

        audio_feat = audio_feat.permute(0,2,1)
        audio_feat = self.audio_encoder1(audio_feat)
        audio_feat = audio_feat.permute(0,2,1)
        

        
        audio_length = [round(i / 4) for i in audio_length]
        audio_mask = sequence_mask(torch.from_numpy(np.asarray(audio_length)), 256).cuda()
        audio_feat, _ = self.audio_rnn(audio_feat, audio_mask)


        audio_feat = audio_feat.permute(0, 2, 1)
        audio_feat = self.audio_encoder2(audio_feat)
        audio_feat = audio_feat.permute(0, 2, 1)


        audio_len = audio_feat.size(1)


        chunked_visual_feat = torch.stack(visual_feat.chunk(chunks=self.chunk_num, dim=-1),
                                          dim=1).view(-1, visual_len, self.model_dim // self.chunk_num)
        

        chunked_audio_feat = torch.stack(audio_feat.chunk(chunks=self.chunk_num, dim=-1),
                                         dim=1).view(-1, audio_len, self.model_dim // self.chunk_num)
        # all in (bs * chunk_num, max_len, model_dim // chunk_num)

        chunked_visual_feat = chunked_visual_feat.transpose(1, 0).contiguous()
        chunked_visual_feat = self.video_self_att(chunked_visual_feat).transpose(1, 0).contiguous()

        # chunked_audio_feat = chunked_audio_feat.transpose(1, 0).contiguous()
        # chunked_audio_feat = self.audio_self_att(chunked_audio_feat).transpose(1, 0).contiguous()

        matrix_a, attn_logit = self.video_audio_attn(chunked_visual_feat, chunked_audio_feat, None)
        # attn_logit in (bs, vis_len, tex_len)
        video_audio_attn, audio_video_attn = attn_logit.softmax(-1), attn_logit.softmax(1)
        matrix_b = video_audio_attn.bmm(audio_video_attn.transpose(-2, -1)).bmm(chunked_visual_feat)
        fusion = torch.cat((chunked_visual_feat, matrix_a, chunked_visual_feat * matrix_a,
                            chunked_visual_feat * matrix_b), dim=-1)
        # (bs, max_len, 1)
        compact_fusion = self.feed_forward(fusion)



        start_features = self.start_rnn(compact_fusion)
        end_features = self.end_rnn(start_features)
        inside_features = self.inside_rnn(compact_fusion)
        

        start_boundary_prob = self.start_block(torch.cat([start_features, compact_fusion], dim=2))
        end_boundary_prob = self.end_block(torch.cat([end_features, compact_fusion], dim=2))
        inside_prob = self.inside_block(torch.cat([inside_features, compact_fusion], dim=2)).squeeze()




        boundary_prob = torch.cat([start_boundary_prob, end_boundary_prob], dim = -1)  #[32*16, 128, 2]
        boundary_prob = boundary_prob.softmax(dim=1)

        boundary_prob = boundary_prob.view(batch_size, self.chunk_num, *boundary_prob.size()[1:]) #[32, 16, 128, 2]


        inside_prob = inside_prob.sigmoid()

        inside_prob = inside_prob.view(batch_size, self.chunk_num, *inside_prob.size()[1:])  #[32, 16, 128]


        return boundary_prob, inside_prob

    def forward(self, visual_feat, textual_feat, audio_feat, textual_mask, audio_mask, audio_length, start, end, epoch):

        real_boundary_prob, real_inside_prob = self.fuse_and_route(visual_feat, textual_feat, audio_feat, textual_mask, audio_mask,audio_length, start, end, epoch)
        # (bs, chunk_num, 1->max_len, 1->task_num)

        avg_boundary_prob = real_boundary_prob.mean(dim=1)
        avg_inside_prob = real_inside_prob.mean(dim=1)
        # assert (avg_boundary_prob.sum(dim=1).allclose(torch.tensor(1.0))), "Invalid Probability!"
        if not self.use_negative:
            return {
                "real_start": avg_boundary_prob[:, :, 0],
                "real_end": avg_boundary_prob[:, :, 1],
                "fake_start": None,
                "fake_end": None,
                "all_start": real_boundary_prob[:, :, :, 0],
                "all_end": real_boundary_prob[:, :, :, 1],
                "inside_prob": avg_inside_prob,
                "order_pred": None,
                "order_target": None,

            }
        else:
            batch_size = visual_feat.size(0)
            idx = list(reversed(range(batch_size)))
            fake_boundary_prob, fake_root_feat= self.fuse_and_route(visual_feat, textual_feat, audio_feat, textual_mask, audio_mask, audio_length)
            return None
