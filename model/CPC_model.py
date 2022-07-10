import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from model.sub_modules.audio_encoder import load_DAVEnet1, load_DAVEnet2, AudioInitializer
from tqdm import tqdm
import numpy as np
# from preprocess import mulaw_decode
import math
from torch.nn import MultiheadAttention
from model.sub_modules.video_encoder import VideoInitializer
from .self_attention import EncoderLayer, Encoder1
from torch import Tensor
#The model is testing


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

#

class Encoder(nn.Module):
    def __init__(self, in_channels, channels, n_embeddings, z_dim, c_dim, video_config):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, channels//2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels//2),
            nn.ReLU(True),
            nn.Conv1d(channels//2, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True)
            )
        self.encoder = nn.Sequential(
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, z_dim),
        )

        self.rnn = nn.GRU(z_dim, c_dim, batch_first=True)
        self.audio_encoder = load_DAVEnet1()
        self.relu = nn.ReLU()
        self.video_fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(True),
            nn.LayerNorm(512),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(True)
        )
        self.video_encoder = VideoInitializer(**video_config)
        self.video_self_att = InternalTemporalRelationModule(input_dim=1024, d_model=1024)
        self.dropout = nn.Dropout(0.2)
        self.cross_att = MultiheadAttention(channels, 4, dropout=0.1)
        self.audio_norm = nn.LayerNorm(channels)


    def encode(self, mels, visual_feat, start, end):
        z = self.conv(mels)
        z = self.encoder(z.transpose(1, 2))
        first_id = 0
        last_id = 64
        visual_feat = self.video_encoder(visual_feat)
        visual_feat = visual_feat.transpose(0,1)
        visual_feat = self.video_self_att(visual_feat).transpose(0,1)
        video_start = start - (start - first_id)
        video_end = end + (last_id - end)
        video_mask = self.Make_mask(start=video_start, end=video_end)
        full_mask = torch.cat([video_mask, video_mask], dim=1)
        concat_video_feat = torch.cat([visual_feat, visual_feat], dim=1)
        concat_video_feat = self.video_fc(concat_video_feat)

        cross_audio_feat =  z

        att_audio_feat = self.cross_att(cross_audio_feat.transpose(0, 1), concat_video_feat.transpose(0, 1),
                                        concat_video_feat.transpose(0, 1),
                                        key_padding_mask=full_mask)[0]
        att_audio_feat = z + self.dropout(att_audio_feat.transpose(0, 1))
        att_audio_feat = self.audio_norm(att_audio_feat)
        c, _ = self.rnn(att_audio_feat)

        return z, c

    def forward(self, mels, visual_feat, start, end, epoch):
        z = self.conv(mels)
        z = self.encoder(z.transpose(1, 2))
        visual_feat = self.video_encoder(visual_feat)
        visual_feat = visual_feat.transpose(0, 1)
        visual_feat = self.video_self_att(visual_feat).transpose(0, 1)
        first_id = 0
        last_id = 64
        batch_size = visual_feat.size(0)

        if epoch <= 20:
            start_distance = 0.0 * (start - first_id) / 10
            start_prob = torch.rand(batch_size).cuda()
            start_distance = start_prob * start_distance
            end_distance = 0.0 * (last_id - end) /10
            end_prob = torch.rand(batch_size).cuda()
            end_distance = end_prob * end_distance
        elif epoch <= 40:
            start_distance = 1.0 * (start - first_id)/10
            start_prob = torch.rand(batch_size).cuda()
            start_distance = start_prob * start_distance
            end_distance = 1.0 * (last_id - end) /10
            end_prob = torch.rand(batch_size).cuda()
            end_distance = end_prob * end_distance
        elif epoch <= 60:
            start_distance = 2.0 * (start - first_id)/10
            start_prob = torch.rand(batch_size).cuda()
            start_distance = start_prob * start_distance
            end_distance = 2.0 * (last_id - end) /10
            end_prob = torch.rand(batch_size).cuda()
            end_distance = end_prob * end_distance
        elif epoch <= 80:
            start_distance = 3.0 * (start - first_id)/10
            start_prob = torch.rand(batch_size).cuda()
            start_distance = start_prob * start_distance
            end_distance = 3.0 * (last_id - end) /10
            end_prob = torch.rand(batch_size).cuda()
            end_distance = end_prob * end_distance
        elif epoch <= 100:
            start_distance = 4.0 * (start - first_id)/10
            start_prob = torch.rand(batch_size).cuda()
            start_distance = start_prob * start_distance
            end_distance = 4.0 * (last_id - end) /10
            end_prob = torch.rand(batch_size).cuda()
            end_distance = end_prob * end_distance
        elif epoch <= 120:
            start_distance = 5.0 * (start - first_id)/10
            start_prob = torch.rand(batch_size).cuda()
            start_distance = start_prob * start_distance
            end_distance = 5.0 * (last_id - end) /10
            end_prob = torch.rand(batch_size).cuda()
            end_distance = end_prob * end_distance
        elif epoch <= 140:
            start_distance = 6.0 * (start - first_id)/10
            start_prob = torch.rand(batch_size).cuda()
            start_distance = start_prob * start_distance
            end_distance = 6.0 * (last_id - end) /10
            end_prob = torch.rand(batch_size).cuda()
            end_distance = end_prob * end_distance
        elif epoch <= 160:
            start_distance = 7.0 * (start - first_id)/10
            start_prob = torch.rand(batch_size).cuda()
            start_distance = start_prob * start_distance
            end_distance = 7.0 * (last_id - end) /10
            end_prob = torch.rand(batch_size).cuda()
            end_distance = end_prob * end_distance
        elif epoch <= 180:
            start_distance = 8.0 * (start - first_id)/10
            start_prob = torch.rand(batch_size).cuda()
            start_distance = start_prob * start_distance
            end_distance = 8.0 * (last_id - end) /10
            end_prob = torch.rand(batch_size).cuda()
            end_distance = end_prob * end_distance
        else:
            start_distance = 10.0 * (start - first_id) / 10
            #start_prob = torch.rand(batch_size).cuda()
            #start_distance = start_prob * start_distance
            end_distance = 10.0 * (last_id - end) / 10
            #end_prob = torch.rand(batch_size).cuda()
            #end_distance = end_prob * end_distance

        start = start - start_distance
        end = end + end_distance

        video_start = start - (start - first_id)
        video_end = end + (last_id - end)


        content_mask = self.Make_mask(start=start, end=end)

        video_mask = self.Make_mask(start=video_start, end=video_end)
        video_mask = ~video_mask
        content_mask = ~content_mask



        full_mask = torch.cat([video_mask, content_mask], dim=1)

        concat_video_feat = torch.cat([visual_feat, visual_feat], dim=1)
        concat_video_feat = self.video_fc(concat_video_feat)

        cross_audio_feat =  z

        att_audio_feat = self.cross_att(cross_audio_feat.transpose(0, 1), concat_video_feat.transpose(0, 1), concat_video_feat.transpose(0, 1),
                       key_padding_mask=full_mask)[0]
        att_audio_feat = z + self.dropout(att_audio_feat.transpose(0, 1))
        att_audio_feat = self.audio_norm(att_audio_feat)
        c, _ = self.rnn(att_audio_feat)

        return z, c

    def Make_mask(self, start, end, first_id=0, last_id=64):


        max_len = last_id
        start_seq = torch.arange(0, max_len).cuda()
        end_seq = torch.arange(0, max_len).cuda()
        batch_size = start.size(0)
        start_seq_expand = start_seq.unsqueeze(0).expand(batch_size, max_len).cuda()
        end_seq_expand = end_seq.unsqueeze(0).expand(batch_size, max_len).cuda()

        start_expand = start.unsqueeze(1).expand_as(start_seq_expand)
        end_expand = end.unsqueeze(1).expand_as(end_seq_expand)
        start_mask = start_seq_expand < start_expand
        start_mask = ~start_mask
        end_mask = end_seq_expand < end_expand

        content_mask = start_mask * end_mask
        return content_mask






class CPC(nn.Module):
    def __init__(self, in_channels, channels, n_embeddings, z_dim, c_dim, n_prediction_steps,video_config, **kwargs):
        super(CPC,self).__init__()
        self.encode = Encoder(in_channels=in_channels,channels=channels, n_embeddings=n_embeddings, z_dim=z_dim, c_dim=c_dim, video_config=video_config)
        self.n_prediction_steps = n_prediction_steps
        self.n_speakers_per_batch = 1

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.Wk = nn.ModuleList([nn.Linear(c_dim, z_dim) for i in range(self.n_prediction_steps)])
        #self.gru = nn.GRU(z_dim, c_dim, batch_first=True)
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim) for _ in range(n_prediction_steps)
        ])
        self.n_negatives = 17
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()



    def forward(self, visual_feat, textual_feat, audio_feat, textual_mask, audio_mask, audio_length, start, end, epoch):
        audio_feat = audio_feat.transpose(2,1)
        batch_size = visual_feat.size(0)

        n_utterances_per_speaker = batch_size
        device = audio_feat.device
        z, c = self.encode(audio_feat, visual_feat, start, end, epoch)
        batch, length, z_dim = z.size()
        target = z[:, -self.n_prediction_steps:, :].transpose(0, 1)

        output = c[:, :-self.n_prediction_steps, :]
        context = output[:, -1, :].view(batch, self.c_dim)
        prediction = torch.empty((self.n_prediction_steps, batch, self.z_dim), dtype=z.dtype, device=device)


        for i in range(self.n_prediction_steps):
            linear = self.Wk[i]
            prediction[i] = linear(context)


        nce = torch.tensor([0],dtype=z.dtype, device=device)


        for i in range(self.n_prediction_steps):
            total = torch.mm(target[i], torch.transpose(prediction[i],0,1))
            total = total.cuda()
            #print("total:",total.dtype)
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0).cuda(), torch.arange(0, batch).cuda()).cuda())  # correct is a tensor
            correct = correct.cuda()
            #print("correct:",total)
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor

        #nce /= -1. * batch * self.n_prediction_steps
        nce /= -1. * self.n_prediction_steps
        accuracy = 1. * (correct.item()) / (batch )

        cpc_loss = nce
        vq_loss = cpc_loss

        return cpc_loss, vq_loss, accuracy



    def nce_loss(self, prediction, target):

        k_size, batch_size, hidden_size = target.size()
        label = torch.arange(0, batch_size * k_size, dtype=torch.long, device=target.device)
        logits = torch.matmul(prediction.reshape(-1, hidden_size),
                              target.reshape(-1, hidden_size).transpose(-1, -2))
        loss = nn.functional.cross_entropy(logits, label, reduction='none')
        accuracy = torch.eq(
            torch.argmax(F.softmax(logits, dim=1), dim=1),
            label)
        nce, acc = [], []
        for i in range(k_size):
            start = i * batch_size
            end = i * batch_size + batch_size
            nce.append(torch.sum(loss[start:end]) / batch_size)
            acc.append(torch.sum(accuracy[start:end], dtype=torch.float) / batch_size)


        return torch.stack(nce).mean(), torch.stack(acc).unsqueeze(0)




