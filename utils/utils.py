#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:12:52
LastEditTime: 2021-08-21 00:16:08
@Description: file content
'''
import os, math, torch, cv2, shutil
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from utils.vgg import VGG
import torch.nn.functional as F

def maek_optimizer(opt_type, cfg, params):
    if opt_type == "ADAM":
        optimizer = torch.optim.Adam(params, lr=cfg['schedule']['lr'], betas=(cfg['schedule']['beta1'], cfg['schedule']['beta2']), eps=cfg['schedule']['epsilon'])
    elif opt_type == "ADAMW":
        optimizer = torch.optim.AdamW(params, lr=cfg['schedule']['lr'], betas=(cfg['schedule']['beta1'], cfg['schedule']['beta2']), eps=cfg['schedule']['epsilon'])
    elif opt_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg['schedule']['lr'], momentum=cfg['schedule']['momentum'])
    elif opt_type == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg['schedule']['lr'], alpha=cfg['schedule']['alpha'])
    else:
        raise ValueError
    return optimizer

def make_loss(loss_type):
    # loss = {}
    if loss_type == "MSE":
        loss = nn.MSELoss(reduction='sum')
    elif loss_type == "L1":
        loss = nn.L1Loss(reduction='sum') # mean
    else:
        raise ValueError
    return loss





def get_path(subdir):
    return os.path.join(subdir)

def save_config(time, log):
    open_type = 'a' if os.path.exists(get_path('../log/' + str(time) + '/records.txt'))else 'w'
    log_file = open(get_path('../log/' + str(time) + '/records.txt'), open_type)
    log_file.write(str(log) + '\n')

def save_net_config(time, log):
    open_type = 'a' if os.path.exists(get_path('../log/' + str(time) + '/net.txt'))else 'w'
    log_file = open(get_path('../log/' + str(time) + '/net.txt'), open_type)
    log_file.write(str(log) + '\n')

def save_net_py(time, py):
    shutil.copyfile(os.path.join('./model', py+'.py'), os.path.join('../log/'+ str(time), py+'.py'))
    
def draw_curve_and_save(x, y, title, filename, precision):
    if not isinstance(x, np.ndarray):
        x = np.array(x).astype(np.int32)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.set_title(title)

    max_y = np.ceil(y.max() / precision) * precision
    min_y = np.floor(y.min() / precision) * precision
    major_y_step = (max_y - min_y) / 10
    if major_y_step < 0.1:
        major_y_step = 0.1
    #设置时间间隔
    ax.yaxis.set_major_locator(MultipleLocator(major_y_step))
    #设置副刻度
    ax.yaxis.set_minor_locator(MultipleLocator(major_y_step))
    ax.yaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='both')
    # ax.legend()
    if (x.shape[0] >= 2):
        axis_range = [x.min(), x.man(), min_y, max_y]
        ax.axis(axis_range)
    ax.plot(x, y)
    plt.savefig(filename)

def calculate_psnr(img1, img2, pixel_range=255, color_mode='y'):
    # transfer color channel to y
    if color_mode == 'rgb':
        img1 = (img1 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
        img2 = (img2 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
    elif color_mode == 'yuv':
        img1 = img1[:, 0, :, :]
        img2 = img2[:, 0, :, :]
    elif color_mode == 'y':
        img1 = img1
        img2 = img2
    # img1 and img2 have range [0, pixel_range]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(pixel_range / math.sqrt(mse))

def ssim(img1, img2, pixel_range=255, color_mode='y'):
    C1 = (0.01 * pixel_range)**2
    C2 = (0.03 * pixel_range)**2

    # transfer color channel to y
    if color_mode == 'rgb':
        img1 = (img1 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
        img2 = (img2 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
    elif color_mode == 'yuv':
        img1 = img1[:, 0, :, :]
        img2 = img2[:, 0, :, :]
    elif color_mode == 'y':
        img1 = img1
        img2 = img2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, pixel_range=255):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2, pixel_range)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2, pixel_range))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), pixel_range)
    else:
        raise ValueError('Wrong input image dimensions.')

class spectra_metric():
    '''
      光谱评价指标 待测图像 x1(shape:[m,n,b]),x2(shape:[m,n,b]);其中m,n为空间分辨率，b为波段数
    '''

    def __init__(self, x1, x2, max_v=1, scale=16):
        self.scale = scale
        self.info = x1.shape
        self.max_v = max_v
        if len(self.info) == 3:
            self.x1_ = x1.reshape([-1, self.info[-1]])
            self.x2_ = x2.reshape([-1, self.info[-1]])
        else:
            self.x1_ = x1
            self.x2_ = x2

    def SAM(self, mode=''):
        t = np.sqrt(np.sum(self.x1_ * self.x1_, axis=-1)) * np.sqrt(np.sum(self.x2_ * self.x2_, axis=-1))
        A = np.sum(self.x1_ * self.x2_, axis=-1)

        A= A/(t)
        _SAM = np.arccos(A) * 180
        _SAM = _SAM / np.pi
        if mode == 'mat':
            return _SAM
        else:
            return np.mean(_SAM)

    def MSE(self, mode=''):
        if mode == 'mat':
            self.MSE_mat = np.mean(np.power(self.x1_ - self.x2_, 2), axis=0)
            return self.MSE_mat
        else:
            return np.mean(np.power(self.x1_ - self.x2_, 2))

    def ERGAS(self):
        k = 100 / self.scale
        return k * np.sqrt(np.mean(self.MSE('mat') / np.power(np.mean(self.x2_, axis=0), 2)))

    def PSNR(self, mode=''):
        _PSNR = 10 * np.log10(np.power(np.max(self.max_v, axis=0), 2) / self.MSE('mat'))
        if mode == 'mat':
            return _PSNR
        else:
            return np.mean(_PSNR)

    def SSIM(self, k1=0.01, k2=0.03, mode=''):
        l = self.max_v
        u1 = np.mean(self.x1_, axis=0).reshape([1, -1])
        u2 = np.mean(self.x2_, axis=0).reshape([1, -1])
        Sig1 = np.std(self.x1_, axis=0).reshape([1, -1])
        Sig2 = np.std(self.x2_, axis=0).reshape([1, -1])
        sig12 = np.sum((self.x1_ - u1) * (self.x2_ - u2), axis=0) / (self.info[0] * self.info[1] - 1)
        c1, c2 = pow(k1 * l, 2), pow(k2 * l, 2)
        SSIM = (2 * u1 * u2 + c1) * (2 * sig12 + c2) / ((u1 ** 2 + u2 ** 2 + c1) * (Sig1 ** 2 + Sig2 ** 2 + c2))
        if mode == 'mat':
            return SSIM
        else:
            return np.mean(SSIM)

    # def get_Evaluation(self,  k1=0.01, k2=0.03):
    #     return self.PSNR(),self.SAM(),self.ERGAS(),self.SSIM(k1=k1, k2=k2),self.MSE()
    #
    # def Evaluation(self,idx=0,k1=0.01,k2=0.03):
    #     PSNR,SAM,ERGAS,SSIM,MSE = self.get_Evaluation(k1,k2)
    #     print(f'{idx}\t{PSNR}\t{SAM}\t{ERGAS}\t{SSIM}\t{np.sqrt(MSE)}')