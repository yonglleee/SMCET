#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2021-12-19 22:56:29
LastEditTime: 2021-12-20 00:28:44
Description: Hyperspectral Pansharpening Using Deep Prior and Dual Attention Residual Network
batch_size = 32, learning_rate = 1e-3, patch_size = 16, L1, ADAM, 1000 epoch, decay 500, x0.1
'''
import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        base_filter = 64
        num_channels = args['data']['n_colors']
        out_channels = args['data']['n_colors']
        self.args = args

        self.conv11 = ConvBlock(num_channels+1, 64, 3, 1, 1, activation='relu', norm=None, bias = True)

        self.conv21 = ConvBlock(64, 64, 3, 1, 1, activation='relu', norm=None, bias = True)
        self.conv22 = ConvBlock(64, 64, 3, 1, 1, activation=None, norm=None, bias = True)
        self.csa2 = csa()

        self.conv31 = ConvBlock(64, 64, 3, 1, 1, activation='relu', norm=None, bias = True)
        self.conv32 = ConvBlock(64, 64, 3, 1, 1, activation=None, norm=None, bias = True)
        self.csa3 = csa()

        self.conv41 = ConvBlock(64, 64, 3, 1, 1, activation='relu', norm=None, bias = True)
        self.conv42 = ConvBlock(64, 64, 3, 1, 1, activation=None, norm=None, bias = True)
        self.csa4 = csa()

        self.conv51 = ConvBlock(64, 64, 3, 1, 1, activation='relu', norm=None, bias = True)
        self.conv52 = ConvBlock(64, 64, 3, 1, 1, activation=None, norm=None, bias = True)
        self.csa5 = csa()

        self.output_conv = ConvBlock(64, out_channels, 5, 1, 2, activation='relu', norm=None, bias = True)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, l_ms, b_ms, x_pan):

        x = torch.cat([b_ms, x_pan], 1)
        x11 = self.conv11(x)

        x21 = self.conv22(self.conv21(x11))
        x22 = self.csa2(x21)
        x = x22 + x11

        x31 = self.conv32(self.conv31(x))
        x32 = self.csa3(x31)
        x = x32 + x

        x41 = self.conv42(self.conv41(x))
        x42 = self.csa4(x41)
        x = x42 + x

        x51 = self.conv52(self.conv51(x))
        x52 = self.csa5(x51)
        x = x52 + x

        x = self.output_conv(x) 

        return x


class csa(nn.Module):
    ''' 
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    def __init__(self):
        super(csa, self).__init__()
        self.ca_module = SELayer(64, 16)
        self.sa_module = ChannelGate2d(64)

    def forward(self, x):
        return self.ca_module(x) + self.sa_module(x)
        
class SELayer(nn.Module):
    ''' 
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelGate2d(nn.Module):
    """
    Channel Squeeze module
    Original paper: https://arxiv.org/abs/1803.02579
    """

    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.Conv2d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: Tensor):  # skipcq: PYL-W0221
        module_input = x
        x = self.squeeze(x)
        x = x.sigmoid()
        return module_input * x

if __name__ == '__main__':

    lms_image = torch.randn(4, 8, 64, 64)
    bms_image = torch.randn(4, 8, 256, 256)
    pan_image = torch.randn(4, 1, 256, 256)
    net = Net(args = '')
    

    y =net(lms_image, bms_image, pan_image)
    print(y.shape)