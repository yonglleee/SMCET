# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = UNetConvBlock(self.split_len2, self.split_len1)
        self.G = UNetConvBlock(self.split_len1, self.split_len2)
        self.H = UNetConvBlock(self.split_len1, self.split_len2)

        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x):
        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = InvBlock(nc, nc // 2)

    def forward(self, x):
        return x + self.block(x)


class FreBlockSpa(nn.Module):
    def __init__(self, nc):
        super(FreBlockSpa, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc))

    def forward(self, x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class FreBlockCha(nn.Module):
    def __init__(self, nc):
        super(FreBlockCha, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1))

    def forward(self, x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class SpatialFuse(nn.Module):
    def __init__(self, in_nc):
        super(SpatialFuse, self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockSpa(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc, in_nc, 3, 1, 1)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 3, 1, 1)

    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        xcat = torch.cat([x, x_freq_spatial], 1)
        x_out = self.cat(xcat)

        return x_out + xori


class ChannelFuse(nn.Module):
    def __init__(self, in_nc):
        super(ChannelFuse, self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockCha(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        xcat = torch.cat([x, x_freq_spatial], 1)
        x_out = self.cat(xcat)

        return x_out + xori


class ProcessBlock(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock, self).__init__()
        self.spa = SpatialFuse(nc)
        self.cha = ChannelFuse(nc)

    def forward(self, x):
        x = self.spa(x)
        x = self.cha(x)

        return x


class ProcessNet(nn.Module):
    def __init__(self, nc):
        super(ProcessNet, self).__init__()
        self.conv0 = nn.Conv2d(nc, nc, 3, 1, 1)
        self.conv1 = ProcessBlock(nc)
        self.downsample1 = nn.Conv2d(nc, nc * 2, stride=2, kernel_size=2, padding=0)
        self.conv2 = ProcessBlock(nc * 2)
        self.downsample2 = nn.Conv2d(nc * 2, nc * 3, stride=2, kernel_size=2, padding=0)
        self.conv3 = ProcessBlock(nc * 3)
        self.up1 = nn.ConvTranspose2d(nc * 5, nc * 2, 1, 1)
        self.conv4 = ProcessBlock(nc * 2)
        self.up2 = nn.ConvTranspose2d(nc * 3, nc * 1, 1, 1)
        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc, nc, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        x34 = self.up1(torch.cat([F.interpolate(x3, size=(x12.size()[2], x12.size()[3]), mode='bilinear'), x12], 1))
        x4 = self.conv4(x34)
        x4 = self.up2(torch.cat([F.interpolate(x4, size=(x01.size()[2], x01.size()[3]), mode='bilinear'), x01], 1))
        x5 = self.conv5(x4)
        xout = self.convout(x5)

        return xout


class InteractNet(nn.Module):
    def __init__(self, in_channel, out_channel, nc):
        super(InteractNet, self).__init__()
        self.extract = nn.Conv2d(in_channel, nc, 1, 1, 0)
        self.process = ProcessNet(nc)
        self.recons = nn.Conv2d(nc, out_channel, 1, 1, 0)

    def forward(self, x):
        x_f = self.extract(x)
        x_f = self.process(x_f) + x_f
        y = self.recons(x_f)

        return y


class Net(nn.Module):
    def __init__(self, in_channel=5, out_channel=4, nc=64,args=''):
        super().__init__()
        in_channel = args['data']['n_colors']+1
        out_channel = args['data']['n_colors']
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 9, 1, 4),
            nn.ReLU()
        )
        self.interact_net = InteractNet(in_channel, out_channel, nc)

    def forward(self, lrms,bms_image, pan):
        # lrms = bms_image
        x = torch.cat([bms_image, pan], dim=1)
        self.conv0(x)
        x = self.interact_net(x)
        return x
    

def prepare_input(resolution):
    lrms = torch.FloatTensor(1, 8, 64, 64)
    pan = torch.FloatTensor(1, 1, 64, 64)
    return dict(lrms=lrms, pan=pan)


if __name__ == '__main__':
    # from ptflops import get_model_complexity_info
    # from thop import profile
    # from fvcore.nn import FlopCountAnalysis, parameter_count_table

    # N = Fourmer(9, 8, 64)

    # ms = torch.randn(1, 8, 64, 64)
    # pan = torch.randn(1, 1, 64, 64)
    # flops, params = profile(N, inputs=(ms, pan))
    # print('Params and FLOPs are {}M and {}G'.format(params / 1e6, flops / 1e9))

    # macs, params = get_model_complexity_info(N, input_res=(1,), input_constructor=prepare_input, as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    # flops = FlopCountAnalysis(N, inputs=(ms, pan))
    # print("FLOPs", flops.total())
    # print(parameter_count_table(N))

    lms_image = torch.randn(4, 4, 64, 64)
    bms_image = torch.randn(4, 4, 256, 256)
    pan_image = torch.randn(4, 1, 256, 256)
    net = Fourmer(5, 4, 64)
    

    y =net(lms_image, bms_image, pan_image)
    print(y.shape)
