#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2021-12-19 15:19:04
LastEditTime: 2021-12-19 16:09:12
Description: SSconv: Explicit Spectral-to-Spatial Convolution for Pansharpening
batch_size = 32, learning_rate = 1e-3, patch_size = 16, L2, ADAM, 600 epoch, decay 200, x0.1
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        base_filter = 64
        ms_channels = args['data']['n_colors']
        pan_channels = 1 
        self.args = args
        spectral_num = ms_channels
        upscale1 = 2
        upscale2 = 4

        self.conv_up1 = SSconv(spectral_num, upscale1)
        self.conv_up2 = SSconv(spectral_num, upscale2)

        self.pan_down = pandown()

        channel_input1 = pan_channels + ms_channels

        channel_output1 = channel_input1 * 4    
        self.conv1 = ConvBlock(channel_input1, channel_output1)
        channel_input2 = channel_output1 + ms_channels + 1  
        self.down1 = Down()
        channel_output2 = channel_output1 * 4    
        self.conv2 = ConvBlock(channel_input2, channel_output2)
        channel_input3 = channel_output2 + ms_channels + 1  #
        self.down2 = Down()
        channel_output3 = channel_input3    
        self.conv3 = ConvBlock(channel_input3, channel_output3)
        channel_input4 = channel_output3 + channel_output2  
        self.up1 = SSconv(channel_output3, 2)
        channel_output4 = 144    
        self.conv4 = ConvBlock(channel_input4, channel_output4)
        channel_input5 = channel_output1 + channel_output4  
        self.up2 = SSconv(channel_output4, 2)
        channel_output5 = 36   
        self.conv5 = ConvBlock(channel_input5, channel_output5)

        self.O_conv3 = ConvBlock(channel_output3, ms_channels, activation=None)
        self.O_conv4 = ConvBlock(channel_output4, ms_channels, activation=None)
        self.O_conv5 = ConvBlock(channel_output5, ms_channels, activation=None)

    def forward(self, l_ms, b_ms, x_pan):

        ms = l_ms
        pan = x_pan
        dim = 1
        panda1, panda2 = self.pan_down(pan)
        ms1 = self.conv_up1(ms)

        ms1 = torch.cat((ms1, panda1), dim)

        ms2 = self.conv_up2(ms)

        ms = torch.cat((ms, panda2), dim)

        x1 = self.conv1(torch.cat((pan, ms2), dim))
        x2 = self.down1(x1)
        x2 = self.conv2(torch.cat((x2, ms1), dim))
        x3 = self.down2(x2)
        x3 = self.conv3(torch.cat((x3, ms), dim))
        x4 = self.up1(x3)
        x4 = self.conv4(torch.cat((x4, x2), dim))
        x5 = self.up2(x4)
        x5 = self.conv5(torch.cat((x5, x1), dim))
        x3 = self.O_conv3(x3)
        x4 = self.O_conv4(x4)
        x5 = self.O_conv5(x5)

        return x5

######################################
#               tool
###################################### 
class SSconv(nn.Module):

    def __init__(self, in_channel, up):
        super().__init__()
        self.conv_up1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * up * up, kernel_size=3,
                                 stride=1, padding=1, bias=True)
        self.up_size = up

    def mapping(self, x):
        B, C, H, W = x.shape
        C1, H1, W1 = C // (self.up_size * self.up_size), H * self.up_size, W * self.up_size
        x = x.reshape(B, C1, self.up_size, self.up_size, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C1, H1, W1)
        return x

    def forward(self, x):
        x = self.conv_up1(x)
        return self.mapping(x)

class pandown(nn.Module):

    def __init__(self):
        super(pandown, self).__init__()
        self.conv_down1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2,
                                          stride=2, padding=0, bias=True)
        self.conv_down2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4,
                                          stride=4, padding=0, bias=True)

        init_weights(self.conv_down1, self.conv_down2)

    def forward(self, pan):
        return self.conv_down1(pan), self.conv_down2(pan)

def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                variance_scaling_initializer(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

def variance_scaling_initializer(tensor):
    # from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor

class Down(nn.Module):

    def __init__(self):
        super().__init__()
        self.max_pool_conv = nn.MaxPool2d(2)

    def forward(self, x):
        return self.max_pool_conv(x)


if __name__ == '__main__':       
    x = torch.randn(1,8,4,4)
    y = torch.randn(1,8,16,16)
    z = torch.randn(1,1,16,16)
    arg = []
    Net = Net(arg)
    out = Net(x, y, z)
    print(out.shape)