#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2022-04-20 11:03:02
LastEditTime: 2022-05-18 21:19:32
Description: HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening 
batch_size = LR_image_size(40/40/64/30), learning_rate = 1e-3, batch_size = 8, L1, ADAM, 1000 epoch, decay 2000, x0.1
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

        self.num_res_blocks = [16, 1, 1, 1, 4]
        self.n_feats = 256
        self.res_scale = 1
        # self.factor = args['data']['upsacle']
        self.factor = 4
        self.out_channels = args['data']['n_colors']
        # FE-PAN & FE-HSI
        # self.LFE_HSI = LFE(in_channels=args['data']['n_colors'])
        self.LFE_HSI = LFE(in_channels=4)
        self.LFE_PAN = LFE(in_channels=1)
        n_head = 4

        ### Multi-Head Attention ###
        lv1_pixels = args['data']['patch_size']**2
        lv2_pixels = (2*args['data']['patch_size'])**2
        lv3_pixels = (4*args['data']['patch_size'])**2

        self.TS_lv3 = MultiHeadAttention(n_head= int(n_head),in_pixels = int(lv1_pixels), linear_dim = int(args['data']['patch_size']), num_features = self.n_feats)
        self.TS_lv2 = MultiHeadAttention(n_head= int(n_head), in_pixels= int(lv2_pixels), linear_dim= int(args['data']['patch_size']), num_features=int(self.n_feats/2))
        self.TS_lv1  = MultiHeadAttention(n_head= int(n_head), in_pixels = int(lv3_pixels), linear_dim = int(args['data']['patch_size']), num_features=int(self.n_feats/4))

        # self.SFE  = SFE(args['data']['n_colors'], self.num_res_blocks[0], self.n_feats, self.res_scale)
        self.SFE  = SFE(args['data']['n_colors'], self.num_res_blocks[0], self.n_feats, self.res_scale)

        ###############
        ### stage11 ###
        ###############
        self.conv11_head = ConvBlock(2*self.n_feats, self.n_feats)
        self.conv12 = ConvBlock(self.n_feats, self.n_feats*2)
        self.ps12 = nn.PixelShuffle(2)

        ###############
        ### stage22 ###
        ###############
        self.conv22_head = ConvBlock(2*int(self.n_feats/2), int(self.n_feats/2))
        self.conv23 = ConvBlock(int(self.n_feats/2), self.n_feats)
        self.ps23 = nn.PixelShuffle(2)


        ###############
        ### stage33 ###
        ###############
        self.conv33_head = ConvBlock(2*int(self.n_feats/4), int(self.n_feats/4))

        ##############
        ### FINAL ####
        ##############
        self.final_conv = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)
    
    def forward(self, l_ms, b_ms, x_pan):
        pan_d = F.interpolate(x_pan, scale_factor=(1/self.factor, 1/self.factor), mode ='bilinear')
        pan_du = F.interpolate(pan_d, scale_factor=(self.factor, self.factor), mode ='bilinear')

        V_lv1, V_lv2, V_lv3 = self.LFE_PAN(x_pan)
        K_lv1, K_lv2, K_lv3 = self.LFE_PAN(pan_du)
        Q_lv1, Q_lv2, Q_lv3 = self.LFE_HSI(b_ms)

        T_lv3  = self.TS_lv3(V_lv3, K_lv3, Q_lv3)
        T_lv2  = self.TS_lv2(V_lv2, K_lv2, Q_lv2)
        T_lv1  = self.TS_lv1(V_lv1, K_lv1, Q_lv1)

        x = self.SFE(l_ms)

        #####################################
        #### stage1: (L/4, W/4) scale ######
        #####################################
        x11 = x
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        x11 = x11 + x11_res

        #####################################
        #### stage2: (L/2, W/2) scale ######
        #####################################
        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))
        #HyperTransformer at (L/2, W/2) scale
        x22_res = x22
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        x22     = x22 + x22_res

        
        #####################################
        ###### stage3: (L, W) scale ########
        #####################################
        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))
        #HyperTransformer at (L, W) scale
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv1), dim=1)
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        x33 = x33 + x33_res


        #####################################
        ############ Feature Pyramid ########
        #####################################
        x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
        x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
        xF  = torch.cat((x11_up, x22_up, x33), dim=1)
        
        #####################################
        ####  Final convolution   ###########
        #####################################
        x = self.final_conv(xF)


        #Final resblocks


        return x

class SFE(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = ConvBlock(in_feats, n_feats)

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResnetBlock(n_feats, scale=res_scale))

        self.conv_tail = ConvBlock(n_feats, n_feats)
    
    def forward(self, x):
        x = self.conv_head(x)
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1 
        return x

class CSFI2(nn.Module):
    def __init__(self, n_feats):
        super(CSFI2, self).__init__()
        self.conv12 = ConvBlock(n_feats, int(n_feats/2), kernel_size=1, stride=1, padding=0)
        self.conv21 = ConvBlock(int(n_feats/2), n_feats, kernel_size=3, stride=2, padding=1)

        self.conv_merge1 = ConvBlock(n_feats*2, n_feats)
        self.conv_merge2 = ConvBlock(n_feats, int(n_feats/2))

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = self.conv12(x12)
        x21 = self.conv12(x2)

        x1 = self.conv_merge1(torch.cat((x1, x21), dim=1))
        x2 = self.conv_merge1(torch.cat((x2, x12), dim=1))

        return x1, x2

class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        
        n_feats1 = n_feats
        self.conv12 = ConvBlock(n_feats1, n_feats1, kernel_size=1, stride=1, padding=0)
        self.conv13 = ConvBlock(n_feats1, n_feats1, kernel_size=1, stride=1, padding=0)

        n_feats2 = int(n_feats/2)
        self.conv21 = ConvBlock(n_feats2, n_feats2, kernel_size=3, stride=2, padding=1)
        self.conv23 = ConvBlock(n_feats2, n_feats2, kernel_size=1, stride=1, padding=0)

        n_feats3 = int(n_feats/4)
        self.conv31_1 = ConvBlock(n_feats2, n_feats2, kernel_size=3, stride=2, padding=1)
        self.conv31_2 = ConvBlock(n_feats2, n_feats2, kernel_size=3, stride=3, padding=1)
        self.conv32 = ConvBlock(n_feats2, n_feats2, kernel_size=3, stride=3, padding=1)

        self.conv_merge1 = ConvBlock(n_feats1*3, n_feats1)
        self.conv_merge2 = ConvBlock(n_feats2*3, n_feats2)
        self.conv_merge3 = ConvBlock(n_feats2*3, n_feats3)
    
    def format(self, x1, x2, x3):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = self.conv12(x12)
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = self.conv13(x13)

        x21 = self.conv21(x2)
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = self.conv23(x23)

        x31 = self.conv31_1(x3)
        x31 = self.conv31_2(x31)
        x32 = self.conv32(x3)

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21, x31), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12, x32), dim=1)))
        x3 = F.relu(self.conv_merge3(torch.cat((x3, x13, x23), dim=1)))
        
        return x1, x2, x3

class MergeTail(nn.Module):
    def __init__(self, n_feats, out_channels):
        super(MergeTail).__init__()
        self.conv13 = ConvBlock(n_feats, int(n_feats/4), kernel_size=1, stride=1, padding=0)
        self.conv23 = ConvBlock(int(n_feats/2), int(n_feats/4), kernel_size=1, stride=1, padding=0)
        self.conv_merge = ConvBlock(3*int(n_feats/4), out_channels)
        self.conv_tail1 = ConvBlock(out_channels, out_channels)
        self.conv_tail2 = ConvBlock(n_feats//2, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = self.conv13(x13)
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = self.conv23(x23)

        x = self.conv_merge(torch.cat((x3, x13, x23), dim=1))
        x = self.conv_tail1(x)
        #x = self.conv_tail2(x)
        return x


# This function implements the learnable spectral feature extractor (abreviated as LSFE)
# Input:    Hyperspectral or PAN image
# Outputs:  out1 = features at original resolution, out2 = features at original resolution/2, out3 = features at original resolution/4
class LFE(nn.Module):
    def __init__(self, in_channels):
        super(LFE, self).__init__()
        #Define number of input channels
        self.in_channels = in_channels
        
        #First level convolutions
        self.conv_64_1  = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn_64_1    = nn.BatchNorm2d(64)
        self.conv_64_2  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_64_2    = nn.BatchNorm2d(64)
        
        #Second level convolutions
        self.conv_128_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_1   = nn.BatchNorm2d(128)
        self.conv_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_2   = nn.BatchNorm2d(128)
        
        #Third level convolutions
        self.conv_256_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn_256_1   = nn.BatchNorm2d(256)
        self.conv_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn_256_2   = nn.BatchNorm2d(256)
        
        #Max pooling
        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #LeakyReLU 
        self.LeakyReLU  = nn.LeakyReLU(negative_slope=0.0)
    def forward(self, x):
        #First level outputs
        out1    = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out1    = self.bn_64_2(self.conv_64_2(out1))

        #Second level outputs
        out1_mp = self.MaxPool2x2(self.LeakyReLU(out1))
        out2    = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out1_mp)))
        out2    = self.bn_128_2(self.conv_128_2(out2))

        #Third level outputs
        out2_mp = self.MaxPool2x2(self.LeakyReLU(out2))
        out3    = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out2_mp)))
        out3    = self.bn_256_2(self.conv_256_2(out3))

        return out1, out2, out3


class LFE_lvx(nn.Module):
    def __init__(self, in_channels, n_feates, level):
        super(LFE_lvx, self).__init__()
        #Define number of input channels
        self.in_channels = in_channels
        self.level = level
        lv1_c = int(n_feates)
        lv2_c = int(n_feates/2)
        lv3_c = int(n_feates/4)

        #First level convolutions
        self.conv_64_1  = nn.Conv2d(in_channels=self.in_channels, out_channels=lv3_c, kernel_size=7, padding=3)
        self.bn_64_1    = nn.BatchNorm2d(lv3_c)
        self.conv_64_2  = nn.Conv2d(in_channels=lv3_c, out_channels=lv3_c, kernel_size=3, padding=1)
        self.bn_64_2    = nn.BatchNorm2d(lv3_c)
        
        #Second level convolutions
        if self.level == 1 or self.level==2:
            self.conv_128_1 = nn.Conv2d(in_channels=lv3_c, out_channels=lv2_c, kernel_size=3, padding=1)
            self.bn_128_1   = nn.BatchNorm2d(lv2_c)
            self.conv_128_2 = nn.Conv2d(in_channels=lv2_c, out_channels=lv2_c, kernel_size=3, padding=1)
            self.bn_128_2   = nn.BatchNorm2d(lv2_c)
        
        #Third level convolutions
        if  self.level == 1:
            self.conv_256_1 = nn.Conv2d(in_channels=lv2_c, out_channels=lv1_c, kernel_size=3, padding=1)
            self.bn_256_1   = nn.BatchNorm2d(lv1_c)
            self.conv_256_2 = nn.Conv2d(in_channels=lv1_c, out_channels=lv1_c, kernel_size=3, padding=1)
            self.bn_256_2   = nn.BatchNorm2d(lv1_c)
        
        #Max pooling
        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #LeakyReLU 
        self.LeakyReLU  = nn.LeakyReLU(negative_slope=0.0)
    def forward(self, x):
        #First level outputs
        out    = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out    = self.bn_64_2(self.conv_64_2(out))

        #Second level outputs
        if self.level == 1 or self.level==2:
            out    = self.MaxPool2x2(self.LeakyReLU(out))
            out    = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out)))
            out    = self.bn_128_2(self.conv_128_2(out))

        #Third level outputs
        if  self.level == 1:
            out     = self.MaxPool2x2(self.LeakyReLU(out))
            out     = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out)))
            out     = self.bn_256_2(self.conv_256_2(out))

        return out

#This function implements the multi-head attention
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn = F.softmax(attn, dim=-1)

        # Attention output
        output  = torch.matmul(attn, v)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, in_pixels, linear_dim, num_features):
        super().__init__()

        self.n_head = n_head
        self.in_pixels = in_pixels
        self.linear_dim = linear_dim

        self.w_qs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)
        self.w_ks = nn.Linear(in_pixels, n_head * linear_dim, bias=False)
        self.w_vs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)
        self.fc = nn.Linear(n_head * linear_dim, in_pixels, bias=False)

        #Scaled dot product attention
        self.attention = ScaledDotProductAttention(temperature=in_pixels ** 0.5)

        #Batch normalization layer
        self.OutBN = nn.BatchNorm2d(num_features=num_features)
    
    def forward(self, v, k, q, mask=None):
        
        b, c, h, w  = q.size(0), q.size(1), q.size(2), q.size(3)
        n_head = self.n_head
        linear_dim = self.linear_dim
        
        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)

        output = v
        # print("q:",q.shape)
        # print("b:", b)
        # print("c:", c)
        # print("n_head:", n_head)
        # print("linear_dim:", linear_dim)



        q = self.w_qs(q).view(b, c, n_head, linear_dim)
        k = self.w_ks(k).view(b, c, n_head, linear_dim)
        v = self.w_vs(v).view(b, c, n_head, linear_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # Computing ScaledDotProduct attention for each head
        v_attn = self.attention(v, k, q, mask=mask)

        v_attn = v_attn.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)
        v_attn = self.fc(v_attn)
        
        
        output = output + v_attn
        #output  = v_attn

        #Reshape output to original image format
        output = output.view(b, c, h, w)

        #We can consider batch-normalization here,,,
        #Will complete it later
        output = self.OutBN(output)
        return output

if __name__ == '__main__':
    x = torch.randn(1,4,4,4)
    y = torch.randn(1,4,16,16)
    z = torch.randn(1,1,16,16)
    arg = []
    Net = Net(arg)
    print(Net)
    out = Net(x, y, z)
    print(out.shape)