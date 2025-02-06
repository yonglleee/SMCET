#!/usr/bin/env python
# coding=utf-8
# -*- coding: utf-8 -*-
"""
File: main.py
Author: LI YONG
Email: your.email@example.com
Github: https://github.com/yourusername
Description: 
"""

from utils.config import get_config
from solver.solver import Solver
import argparse
import torch

if __name__ == '__main__':

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser(description='Pansharpening')
    parser.add_argument('--option_path', type=str, default='yml/IKONOS/mgct_fdnoconv3_conv3_1111_attmmp.yml')
    parser.add_argument('--use_kfold', action='store_true', help='Whether to use K-fold cross-validation')
    parser.add_argument('--fold', type=int, default=0, help='The current fold number for cross-validation (0-indexed)')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)

        # 更新配置
    cfg['use_kfold'] = opt.use_kfold
    cfg['fold'] = opt.fold

    print(cfg)
    solver = Solver(cfg)
    solver.run()
    