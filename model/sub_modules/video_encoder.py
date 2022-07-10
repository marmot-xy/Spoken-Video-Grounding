import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, stride, padding):
        super(TCN, self).__init__()

        self.hidden_dim = hidden_dim//4
        self.conv1_2 = nn.Conv1d(in_channels=input_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv1d(in_channels=input_dim, out_channels=self.hidden_dim, kernel_size=5, padding=2)
        self.conv1_4 = nn.Conv1d(in_channels=input_dim, out_channels=self.hidden_dim, kernel_size=7, padding=3)
        self.conv1_5 = nn.Conv1d(in_channels=input_dim, out_channels=self.hidden_dim, kernel_size=1,
                                 padding=0)

    def forward(self, input:Tensor):
        y = input.permute(0, 2, 1)
        out_2 = self.conv1_2(y)
        out_3 = self.conv1_3(y)
        out_4 = self.conv1_4(y)
        out_5 = self.conv1_5(y)

        out_2 = out_2.permute(0, 2, 1)
        out_3 = out_3.permute(0, 2, 1)
        out_4 = out_4.permute(0, 2, 1)
        out_5 = out_5.permute(0, 2, 1)

        out = torch.cat([out_2, out_3, out_4, out_5], 2)  # test
        return out


def _construct_conv_layers(input_dim, hidden_dim, kernel_size, stride, padding):
    layers = []
    for layer_idx in range(len(kernel_size)):
        in_dim = input_dim if layer_idx == 0 else hidden_dim
        out_dim = hidden_dim
        layers.append(nn.Conv1d(in_dim, out_dim, kernel_size[layer_idx],
                                stride[layer_idx], padding[layer_idx]))
        layers.append(nn.GELU())
        layers.append(nn.BatchNorm1d(out_dim))
    return nn.Sequential(*layers)


class VideoInitializer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.init_conv = _construct_conv_layers(input_dim, hidden_dim, kernel_size, stride, padding)

    def forward(self, visual_feat):
        return self.init_conv(visual_feat.transpose(-2, -1)).transpose(-2, -1)


class VideoFusionInitializer(nn.Module):
    def __init__(self, frame_dim, motion_dim, hidden_dim, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.frame_conv = nn.Sequential(
            nn.Conv1d(frame_dim, hidden_dim, kernel_size[0], stride[0], padding[0]),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.motion_conv = nn.Sequential(
            nn.Conv1d(motion_dim, hidden_dim, kernel_size[0], stride[0], padding[0]),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, visual_feat):
        frame_raw_feat, motion_raw_feat = torch.chunk(visual_feat, dim=-1, chunks=2)
        frame_feat = self.frame_conv(frame_raw_feat.transpose(-2, -1)).transpose(-2, -1)
        motion_feat = self.motion_conv(motion_raw_feat.transpose(-2, -1)).transpose(-2, -1)
        return F.gelu(frame_feat + motion_feat) - (frame_feat - motion_feat) ** 2


class VideoSeparateInitializer(nn.Module):
    def __init__(self, frame_dim, motion_dim, hidden_dim, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.frame_conv = nn.Sequential(
            nn.Conv1d(frame_dim, hidden_dim, kernel_size[0], stride[0], padding[0]),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.motion_conv = nn.Sequential(
            nn.Conv1d(motion_dim, hidden_dim, kernel_size[0], stride[0], padding[0]),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, visual_feat):
        frame_raw_feat, motion_raw_feat = torch.chunk(visual_feat, dim=-1, chunks=2)
        frame_feat = self.frame_conv(frame_raw_feat.transpose(-2, -1)).transpose(-2, -1)
        motion_feat = self.motion_conv(motion_raw_feat.transpose(-2, -1)).transpose(-2, -1)
        return frame_feat, motion_feat



