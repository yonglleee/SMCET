#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-17 22:19:38
LastEditTime: 2021-01-19 17:13:17
@Description: file content
'''
from solver.basesolver import BaseSolver
import os, torch, time, cv2, importlib
import torch.backends.cudnn as cudnn
from data.data import *
from torch.utils.data import DataLoader
from torch.autograd import Variable 
import numpy as np
from PIL import Image
import rasterio
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from thop import profile

class Testsolver(BaseSolver):
    def __init__(self, cfg):
        super(Testsolver, self).__init__(cfg)
        
        self.net_name = self.cfg['algorithm'].lower()
        self.net_type = self.cfg['type'].lower()
        lib = importlib.import_module('model.' + self.net_name)
        net = lib.Net
        
        self.model = net(
            args = self.cfg
        )

    def check(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
              
            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >=0:
                    self.gpu_ids.append(gid)
            torch.cuda.set_device(self.gpu_ids[0]) 
            
            # self.model_path = os.path.join(self.cfg['checkpoint'], self.cfg['data_name'],self.cfg['algorithm'], self.cfg['test']['model'])
            # self.model_path = os.path.join(self.cfg['checkpoint'], self.cfg['data_name'],self.cfg['algorithm'], self.cfg['test']['model'])
            self.model_path = os.path.join(self.cfg['test']['model'])
            self.model = self.model.cuda(self.gpu_ids[0])
            # self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
            static = torch.load(self.model_path, map_location=lambda storage, loc: storage)['net']
            self.model.load_state_dict(static)

    def test(self):
        self.model.eval()
        avg_time= []
        for i, batch in enumerate(self.data_loader):
            # if i >= 2:
            #     break

            ms_image, lms_image, pan_image, bms_image, name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])
            # print(ms_image.shape)
            if self.cuda:
                ms_image = ms_image.cuda(self.gpu_ids[0])
                lms_image = lms_image.cuda(self.gpu_ids[0])
                pan_image = pan_image.cuda(self.gpu_ids[0])
                bms_image = bms_image.cuda(self.gpu_ids[0])
            # print(ms_image.shape)
            t0 = time.time()
            with torch.no_grad():
                # print(pan_image.shape)
                prediction = self.model(lms_image, bms_image, pan_image)

                # flops, params = profile(self.model, inputs=(lms_image, bms_image, pan_image))
                # flops_m = flops / 1_000_000
                # params_m = params / 1_000_000
                # a_name = self.cfg['algorithm']

                # print(f'{a_name}:FLOPs: {flops_m:.1f} M, Parameters: {params_m:.3f} M')

            t1 = time.time()

            if self.cfg['data']['normalize'] :
                ms_image = (ms_image+1) /2
                lms_image = (lms_image+1) /2
                pan_image = (pan_image+1) /2
                bms_image = (bms_image+1) /2

            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            avg_time.append(t1 - t0)


            self.save_img(bms_image.cpu().data, name[0][0:-4]+'_bic.tif', mode='MS')
            self.save_img(ms_image.cpu().data, name[0][0:-4]+'_gt.tif', mode='MS')
            self.save_img(prediction.cpu().data, name[0][0:-4]+'.tif', mode='MS')
            self.save_img(pan_image.cpu().data, name[0][0:-4]+'_pan.tif', mode='PAN')
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time[1:])))
        
    def eval(self):
        self.model.eval()
        avg_time= []
        for batch in self.data_loader:

            ms_image, lms_image, pan_image, bms_image, name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]),  Variable(batch[3]), (batch[4])
            if self.cuda:
                lms_image = lms_image.cuda(self.gpu_ids[0])
                pan_image = pan_image.cuda(self.gpu_ids[0])
                bms_image = bms_image.cuda(self.gpu_ids[0])

            t0 = time.time()
            with torch.no_grad():
                prediction = self.model(lms_image, bms_image, pan_image)  
            t1 = time.time()

            if self.cfg['data']['normalize'] :
                lms_image = (lms_image+1) /2
                pan_image = (pan_image+1) /2
                bms_image = (bms_image+1) /2

            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            avg_time.append(t1 - t0)
            self.save_img(lms_image.cpu().data, name[0][0:-4] + '_lms.tif', mode='MS')
            self.save_img(bms_image.cpu().data, name[0][0:-4]+'_bic.tif', mode='MS')
            self.save_img(prediction.cpu().data, name[0][0:-4]+'.tif', mode='MS')
            self.save_img(pan_image.cpu().data, name[0][0:-4]+'_pan.tif', mode='PAN')
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))

    def save_img(self, img, img_name, mode):
        save_img = img.squeeze(dim=0).clamp(0, 1).numpy()
        save_img = save_img[:4,:,:]
        # print(save_img.shape)
        # save img
        # test_low =False
        if self.cfg['test']['type'] == 'test':
            save_dir=os.path.join('../results/',self.cfg['data_name'],self.cfg['algorithm'],self.cfg['test']['type'],self.cfg['schedule']['loss']+'_auxi_lambda_'+str(self.cfg['schedule']['auxi_lambda']))
            
        else:
            save_dir = os.path.join('../results/', self.cfg['data_name'], self.cfg['algorithm'],
                                    self.cfg['test']['type']+'_full',self.cfg['schedule']['loss']+'_auxi_lambda_'+str(self.cfg['schedule']['auxi_lambda']))
        # print(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_fn = save_dir +'/'+ img_name

        save_img = np.uint8(save_img*255)

        if mode=='MS':
            with rasterio.open(save_fn, 'w', driver='GTiff', height=save_img.shape[1], width=save_img.shape[2], count=4, dtype='uint8') as dst:
                dst.write(save_img)
        elif mode=='PAN':
            with rasterio.open(save_fn, 'w', driver='GTiff', height=save_img.shape[1], width=save_img.shape[2], count=1, dtype='uint8') as dst:
                dst.write(save_img)



        # save_img = Image.fromarray(save_img, mode='RGBA')
        # #
        # save_img.save(save_fn, format='TIFF')

    def gaussian_batch(self, dim):
        if self.cfg['gpu_mode']:
            return torch.randn(tuple(dim)).to(self.cfg['gpus'][0])
        else:
            return torch.randn(tuple(dim))
  
    def run(self):
        self.check()
        if self.cfg['test']['type'] == 'test':            
            self.dataset = get_test_data(self.cfg, self.cfg['test']['data_dir'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
                num_workers=self.cfg['threads'])
            self.test()
        elif self.cfg['test']['type'] == 'eval':            
            self.dataset = get_eval_data(self.cfg, self.cfg['test']['data_dir'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
                num_workers=self.cfg['threads'])
            self.eval()
        else:
            raise ValueError('Mode error!')