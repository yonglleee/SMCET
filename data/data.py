#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-16 19:22:41
LastEditTime: 2020-12-23 14:51:28
@Description: file content
'''
from os.path import join
from torchvision.transforms import Compose, ToTensor
#from .dataset2 import Data, Data_test, Data_eval
from .dataset3 import Data, Data_test, Data_eval
from torchvision import transforms
import torch,  numpy
import torch.utils.data as data

def transform():
    return Compose([
        ToTensor(),
    ])
    
def get_data(cfg, mode):
    data_dir_ms = join(mode, cfg['source_ms'])
    data_dir_pan = join(mode, cfg['source_pan'])
    data_dir_mtf = join(mode, cfg['data']['source_mtf'])
    cfg = cfg
    return Data(data_dir_ms, data_dir_pan, data_dir_mtf, cfg, transform=transform())
    
def get_test_data(cfg, mode):
    data_dir_ms = join(mode, cfg['test']['source_ms'])
    data_dir_pan = join(mode, cfg['test']['source_pan'])
    data_dir_mtf = join(mode, cfg['data']['source_mtf'])
    cfg = cfg
    return Data_test(data_dir_ms, data_dir_pan, data_dir_mtf, cfg, transform=transform())

def get_eval_data(cfg, mode):
    data_dir_ms = join(mode, cfg['test']['source_ms'])
    data_dir_pan = join(mode, cfg['test']['source_pan'])
    data_dir_mtf = join(mode, cfg['data']['source_mtf'])
    cfg = cfg
    return Data_eval(data_dir_ms, data_dir_pan, data_dir_mtf, cfg, transform=transform())