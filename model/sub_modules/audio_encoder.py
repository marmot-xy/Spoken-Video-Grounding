import math
import torch.nn as nn
import numpy as np
import torch

from modules import DynamicGRU
def conv1x9(in_planes, out_planes, stride=1):
    """1x9 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,9), stride=stride, padding=(0,4), bias=False)

def conv1d(in_planes, out_planes, width=9, stride=1, bias=False):
    """1xd convolution with padding"""
    if width % 2 == 0:
        pad_amt = int(width / 2)
    else:
        pad_amt = int((width - 1) / 2)
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,width), stride=stride, padding=(0,pad_amt), bias=bias)

class SpeechBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, width=9, stride=1, downsample=None):
        super(SpeechBasicBlock, self).__init__()
        self.conv1 = conv1d(inplanes, planes, width=width, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(planes, planes, width=width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResDavenet(nn.Module):
    def __init__(self, feat_dim=80, block=SpeechBasicBlock, layers=[2, 2, 2, 2], layer_widths=[128, 128, 256, 512, 1024], convsize=9):
        super(ResDavenet, self).__init__()
        self.feat_dim = feat_dim
        self.inplanes = layer_widths[0]
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=(self.feat_dim,1), stride=1, padding=(0,0), bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, layer_widths[1], layers[0], width=convsize, stride=2)
        self.layer2 = self._make_layer(block, layer_widths[2], layers[1], width=convsize, stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, width=9, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, width=width, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, width=width, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.squeeze(2)
        return x

def load_DAVEnet1():
    layer_widths = [128,128,256]
    layer_depths = [1,1]
    audio_model = ResDavenet(feat_dim=128, layers=layer_depths, convsize=9, layer_widths=layer_widths)
    return audio_model

def load_DAVEnet2():
    layer_widths = [256, 512, 1024]
    layer_depths = [2, 2]
    audio_model = ResDavenet(feat_dim=256, layers=layer_depths, convsize=9, layer_widths=layer_widths)

    return audio_model


class AudioInitializer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = DynamicGRU(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, textual_feat, textual_mask):
        textual_len = textual_mask.sum(-1) if textual_mask is not None else None
        return self.gru(textual_feat, textual_len)




def Make_mask(start, end, first_id=0, last_id=64):


        max_len = last_id
        start_seq = torch.arange(0, max_len)
        end_seq = torch.arange(0, max_len)
        batch_size = start.size(0)
        start_seq_expand = start_seq.unsqueeze(0).expand(batch_size, max_len)
        end_seq_expand = end_seq.unsqueeze(0).expand(batch_size, max_len)

        start_expand = start.unsqueeze(1).expand_as(start_seq_expand)
        end_expand = end.unsqueeze(1).expand_as(end_seq_expand)
        start_mask = start_seq_expand < start_expand
        start_mask = ~start_mask
        end_mask = end_seq_expand < end_expand

        content_mask = start_mask * end_mask
        return content_mask


