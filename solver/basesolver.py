#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:07:03
LastEditTime: 2021-08-21 00:32:21
@Description: file content
'''
import os, torch, time
from utils.utils import  draw_curve_and_save, save_config
from data.dataset import data
from data.data import get_data,get_test_data
from torch.utils.data import DataLoader

class BaseSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nEpochs = cfg['nEpochs']
        self.checkpoint_dir = cfg['checkpoint']
        self.epoch = 1

        time_now = int(time.time())
        time_local = time.localtime(time_now)
        self.timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time_local)

        if cfg['gpu_mode']:
            self.num_workers = cfg['threads']
        else:
            self.num_workers = 0

        self.train_dataset = get_data(cfg, cfg['data_dir_train'])
        self.train_loader = DataLoader(self.train_dataset, cfg['data']['batch_size'], shuffle=False,
            num_workers=self.num_workers)
        self.val_dataset = get_data(cfg, cfg['data_dir_eval'])
        self.val_loader = DataLoader(self.val_dataset, cfg['data']['batch_size'], shuffle=False,
            num_workers=self.num_workers)
        self.test_dataset = get_test_data(self.cfg, self.cfg['test']['data_dir'])
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=1,
                                      num_workers=self.cfg['threads'])

        self.records = {'Epoch': [], 'PSNR': [], 'SSIM': [], 'EGRS': [],'Loss': []}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            self.epoch = ckpt['epoch']
            self.records = ckpt['records']
        else:
            raise FileNotFoundError

    def save_checkpoint(self):
        self.ckp = {
            'epoch': self.epoch,
            'records': self.records,
        }

    def train(self):
        raise NotImplementedError
    
    def eval(self):
        raise NotImplementedError
    
    def run(self):
        while self.epoch <= self.nEpochs:
            self.train()
            self.eval()
            self.save_checkpoint()
            # self.save_records()
            self.epoch += 1
        #self.logger.log('Training done.')
