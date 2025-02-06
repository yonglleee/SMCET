#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:04:48
LastEditTime: 2021-08-21 00:05:46
@Description: file content
'''
import os, importlib, torch, shutil, time


from solver.basesolver import BaseSolver
from utils.utils import maek_optimizer, make_loss, calculate_psnr, calculate_ssim, save_config, save_net_config, save_net_py, spectra_metric
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from torch.optim import lr_scheduler
from utils.lr_scheduler import CosineAnnealingRestartCyclicLR,WarmupHalfCycleCosine
from importlib import import_module
from torch.autograd import Variable
from torch.utils.data import DataLoader,Subset
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils.config import save_yml
import rasterio

from sklearn.model_selection import KFold

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch import autograd



class Solver(BaseSolver):
    def __init__(self, cfg):
        super(Solver, self).__init__(cfg)
        self.savedAcc = []
        self.savedmodels = []
        self.savedNum = 4
        self.init_epoch = self.cfg['schedule']
        self.auxi_lambda = self.cfg['schedule']['auxi_lambda']
        self.auxi_mode = "fft"
        self.auxi_loss = self.cfg['schedule']['loss']  #"L1 MSE"
        self.module_first = 1
        
        self.net_name = self.cfg['algorithm'].lower()
        self.net_type= self.cfg['type'].lower()

        lib = importlib.import_module('model.' + self.net_name)
        net = lib.Net

        self.model = net(
            args = self.cfg
        )

        self.optimizer = maek_optimizer(self.cfg['schedule']['optimizer'], cfg, self.model.parameters())


        self.milestones = list(map(lambda x: int(x), self.cfg['schedule']['decay'].split('-')))
        if self.cfg['type'].lower() == 'mgct':
            self.scheduler = CosineAnnealingRestartCyclicLR(self.optimizer,periods=self.cfg['schedule']['periods'],restart_weights=self.cfg['schedule']['restart_weights'],eta_mins=self.cfg['schedule']['eta_mins'])
            #self.scheduler= WarmupHalfCycleCosine(self.optimizer,warmup_epochs=100,total_epochs=cfg['nEpochs'], max_lr=cfg['schedule']['lr'] )
        else:
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, self.milestones,
                                                      gamma=self.cfg['schedule']['gamma'], last_epoch=-1)
        self.loss = make_loss(self.cfg['schedule']['loss'])
        # if self.net_type == 'psinn':
        #     self.loss2 = make_loss(self.cfg['schedule']['bloss'])

        self.log_name = self.cfg['data_name']+'/'+self.cfg['algorithm'] + '/'+self.cfg['algorithm'] + '_' + str(self.cfg['data']['upsacle']) + '_loss_'+str(self.cfg['schedule']['loss']) + '_lambda_'+str(str(self.cfg['schedule']['auxi_lambda']) )+'_' + str(self.timestamp)
        # save log
        self.writer = SummaryWriter('../log/'+'/' + str(self.log_name))
        save_net_config(self.log_name, self.model)
        save_net_py(self.log_name, self.net_name)
        save_yml(cfg, os.path.join('../log/' + str(self.log_name), 'config.yml'))
        save_config(self.log_name, 'Train dataset has {} images and {} batches.'.format(len(self.train_dataset), len(self.train_loader)))
        save_config(self.log_name, 'Val dataset has {} images and {} batches.'.format(len(self.val_dataset), len(self.val_loader)))
        save_config(self.log_name, 'Model parameters: '+ str(sum(param.numel() for param in self.model.parameters())))

    def train(self): 
        with tqdm(total=len(self.train_loader), miniters=1,
                desc='Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t:

            epoch_loss = 0
            for iteration, batch in enumerate(self.train_loader, 1):
                ms_image, lms_image, pan_image, bms_image, file= Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])

                if self.cuda:
                    ms_image, lms_image, pan_image, bms_image  = ms_image.cuda(self.gpu_ids[0]), lms_image.cuda(self.gpu_ids[0]), pan_image.cuda(self.gpu_ids[0]), bms_image.cuda(self.gpu_ids[0])

                self.optimizer.zero_grad()               
                self.model.train()

                y = self.model(lms_image, bms_image, pan_image)
                #print(y.shape)

                loss_m=loss_auxi=0
                if self.auxi_lambda:
                    
                    # fft shape: [B, P, D]
                    if self.auxi_mode == "fft":
                        # loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)
                        loss_auxi = torch.fft.fft2(y) - torch.fft.fft2(ms_image)

                    # elif self.args.auxi_mode == "rfft":
                    #     loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
                    # else:
                    #     raise NotImplementedError

                    if self.auxi_loss == "L1":
                        # MAE, 最小化element-wise error的模长
                        loss_auxi = loss_auxi.abs().mean() if self.module_first else loss_auxi.mean().abs()  # check the dim of fft
                    elif self.auxi_loss == "MSE":
                        # MSE, 最小化element-wise error的模长
                        loss_auxi = (loss_auxi.abs()**2).mean() if self.module_first else (loss_auxi**2).mean().abs()
                    else:
                        raise NotImplementedError

                    # loss_auxi = self.auxi_lambda * loss_auxi
                
                # else:
                #     loss_m = self.loss(y, ms_image) / (self.cfg['data']['batch_size'] * 2)
            
                loss_m = self.loss(y, ms_image)
                
                loss = (1-self.auxi_lambda)*loss_m + self.auxi_lambda * loss_auxi 


                epoch_loss += loss.data
                t.set_postfix_str("Batch loss {:.4f}".format(loss.item()))
                t.update()

                loss.backward()
                # print("grad before clip:"+str(self.model.output_conv.conv.weight.grad))
                # for param in self.model.parameters():
                #     if param.grad != None:
                #         print("param", param.grad.item())
                if self.cfg['schedule']['gclip'] > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg['schedule']['gclip']
                    )
                self.optimizer.step()
                
            self.records['Loss'].append(epoch_loss / len(self.train_loader))

            # ms_image1 = ms_image[0]
            # y1 = y[0]
            # lms_image1 = lms_image[0]

            # self.writer.add_image('HR MS image', ms_image1[0:3,:,:], self.epoch)
            # self.writer.add_image('Fused image', y1[0:3,:,:], self.epoch)

            # self.writer.add_image('MS image', lms_image1[0:3,:,:], self.epoch)
            save_config(self.log_name, 'Initial Training Epoch {}: Loss={:.4f}'.format(self.epoch, self.records['Loss'][-1]))
            self.writer.add_scalar('Loss_epoch', self.records['Loss'][-1], self.epoch)
            self.writer.add_scalar('lr_epoch', self.scheduler.get_last_lr()[0] , self.epoch)


    def eval(self):
        with tqdm(total=len(self.val_loader), miniters=1,
                desc='Val Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t1:
            psnr_list, ssim_list, egrs_list= [], [], []
            for iteration, batch in enumerate(self.val_loader, 1):
                
                ms_image, lms_image, pan_image, bms_image, file = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])
                if self.cuda:
                    ms_image, lms_image, pan_image, bms_image = ms_image.cuda(self.gpu_ids[0]), lms_image.cuda(self.gpu_ids[0]), pan_image.cuda(self.gpu_ids[0]), bms_image.cuda(self.gpu_ids[0])

                self.model.eval()
                with torch.no_grad():


                    y = self.model(lms_image, bms_image, pan_image)
                    # print(y.shape)
                    loss = self.loss(y, ms_image) / (self.cfg['data']['batch_size'] * 2)

                batch_psnr, batch_ssim,batch_egrs = [], [], []
                y = y
                ms_image=ms_image
                for c in range(y.shape[0]):
                    if not self.cfg['data']['normalize']:
                        predict_y = (y[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        ground_truth = (ms_image[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                    else:          
                        predict_y = (y[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        ground_truth = (ms_image[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5

                    Metric = spectra_metric(ground_truth, predict_y, max_v=255, scale=4)
                    psnr = Metric.PSNR()
                    ssim = Metric.SSIM()
                    egrs = Metric.ERGAS()
                    # psnr = calculate_psnr(predict_y, ground_truth, 255)
                    # ssim = calculate_ssim(predict_y, ground_truth, 255)
                    batch_psnr.append(psnr)
                    batch_ssim.append(ssim)
                    batch_egrs.append(egrs)
                avg_psnr = np.array(batch_psnr).mean()
                avg_ssim = np.array(batch_ssim).mean()
                avg_egrs = np.array(batch_egrs).mean()
                psnr_list.extend(batch_psnr)
                ssim_list.extend(batch_ssim)
                egrs_list.extend(batch_egrs)
                t1.set_postfix_str('Batch loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, EGRS: {:.4f}'.format(loss.item(), avg_psnr, avg_ssim, avg_egrs))
                t1.update()
            self.records['Epoch'].append(self.epoch)
            self.records['PSNR'].append(np.array(psnr_list).mean())
            self.records['SSIM'].append(np.array(ssim_list).mean())
            self.records['EGRS'].append(np.array(egrs_list).mean())

            save_config(self.log_name, 'Val Epoch {}: PSNR={:.4f}, SSIM={:.4f},EGRS={:.4f}'.format(self.epoch, self.records['PSNR'][-1],
                                                                    self.records['SSIM'][-1],  self.records['EGRS'][-1],))
            self.writer.add_scalar('PSNR_epoch', self.records['PSNR'][-1], self.epoch)
            self.writer.add_scalar('SSIM_epoch', self.records['SSIM'][-1], self.epoch)
            self.writer.add_scalar('EGRS_epoch', self.records['EGRS'][-1], self.epoch)
    def test(self):
        self.model.eval()
        avg_time= []
        for batch in self.test_loader:

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
                if self.net_type == 'psinn':
                    # forward
                    y = self.model(lms_image, bms_image, pan_image, ms_image)
                    hf_ = torch.cat((lms_image, self.gaussian_batch(y[:, 4:, :, :].shape)), 1)
                    y_rev = self.model(hf_, bms_image, pan_image, ms_image, rev=True)


                    prediction = y_rev

                else:
                    # print(pan_image.shape)
                    prediction = self.model(lms_image, bms_image, pan_image)
            t1 = time.time()

            if self.cfg['data']['normalize'] :
                ms_image = (ms_image+1) /2
                lms_image = (lms_image+1) /2
                pan_image = (pan_image+1) /2
                bms_image = (bms_image+1) /2

            # print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            avg_time.append(t1 - t0)


            self.save_img(bms_image.cpu().data, name[0][0:-4]+'_bic.tif', mode='CMYK')
            self.save_img(ms_image.cpu().data, name[0][0:-4]+'_gt.tif', mode='CMYK')
            self.save_img(prediction.cpu().data, name[0][0:-4]+'.tif', mode='CMYK')
            # self.save_img(pan_image.cpu().data, name[0][0:-4]+'_pan.tif', mode='L')
        # print("===> AVG Timer: %.4f sec." % (np.mean(avg_time[1:])))

    def save_img(self, img, img_name, mode):
        save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
        # save img
        save_dir = os.path.join('../results/', self.cfg['data_name'], self.cfg['test']['algorithm'],
                                self.cfg['test']['type'])
        # print(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_fn = save_dir + '/' + img_name

        save_img = np.uint8(save_img * 255).astype('uint8').transpose(2, 0, 1)
        with rasterio.open(save_fn, 'w', driver='GTiff', height=save_img.shape[1], width=save_img.shape[2], count=4,
                           dtype='uint8') as dst:
            dst.write(save_img)

    def check_gpu(self):
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
            self.loss = self.loss.cuda(self.gpu_ids[0])

            self.model = self.model.cuda(self.gpu_ids[0])
            # self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)

    def check_pretrained(self):
        checkpoint = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['data_name'],self.cfg['algorithm'], self.cfg['pretrain']['pre_sr'])

        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['net'])
            self.epoch = torch.load(checkpoint, map_location=lambda storage, loc: storage)['epoch']
            for _ in range(self.epoch-1): self.scheduler.step()
            self.optimizer.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['optimizer'])
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must less than the max epoch!")
        else:
            print(checkpoint)
            raise Exception("Pretrain path error!")

    def save_checkpoint(self):
        super(Solver, self).save_checkpoint()
        self.ckp['net'] = self.model.state_dict()
        self.ckp['optimizer'] = self.optimizer.state_dict()
        root_checkpoint_pth = '../checkpoint'
        if not os.path.exists(root_checkpoint_pth + '/' + str(self.log_name)):
            os.makedirs(root_checkpoint_pth + '/' + str(self.log_name))
        torch.save(self.ckp, os.path.join(root_checkpoint_pth + '/' + str(self.log_name), 'latest.pth'))

        # if self.cfg['save_best']:
        #     if self.records['SSIM'] != [] and self.records['SSIM'][-1] == np.array(self.records['SSIM']).max():
        #         shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
        #                     os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'best.pth'))
        #         # self.test()
        #         print("best")
        if self.cfg['save_best']:
            accu = self.records['SSIM'][-1]
            out_file = str(accu) + '_' + str(self.epoch)
            if len(self.savedmodels) == self.savedNum:
                minAcc = min(self.savedAcc)
                minAccIndex = self.savedAcc.index(minAcc)
                if accu > minAcc:
                    temAcc = self.savedAcc.pop(minAccIndex)
                    tmpfile = self.savedmodels.pop(minAccIndex)
                    os.remove(tmpfile)
                else:
                    return
            self.savedmodels.append(os.path.join(root_checkpoint_pth + '/' + str(self.log_name), out_file + ".pth"))
            self.savedAcc.append(accu)

            shutil.copy(os.path.join(root_checkpoint_pth + '/' + str(self.log_name), 'latest.pth'),
                                 os.path.join(root_checkpoint_pth + '/' + str(self.log_name), out_file + ".pth"))



    def gaussian_batch(self, dim):
        if self.cfg['gpu_mode']:
            return torch.randn(tuple(dim)).to(self.cfg['gpus'][0])
        else:
            return torch.randn(tuple(dim))


    def run(self):
        print("torch.backends.cuda.matmul.allow_tf32",torch.backends.cuda.matmul.allow_tf32 )
        self.check_gpu()
        if self.cfg['pretrain']['pretrained']:
            self.check_pretrained()
        # Resume...
        # elif self.cfg['resume'] is not None:
        #     print("Loading from existing FCN and copying weights to continue....")
        #     checkpoint = torch.load(self.cfg['resume'], map_location=lambda storage, loc: storage)['net']
        #
        #     self.model.load_state_dict(checkpoint, strict=False)
        try:
            while self.epoch <= self.nEpochs:
                # with autograd.detect_anomaly():
                self.train()
                if self.epoch >1000:
                    self.eval()
                    self.save_checkpoint()
                elif self.epoch <=1000 and self.epoch%5 ==0:
                    self.eval()
                    self.save_checkpoint()
                self.scheduler.step()
                self.epoch += 1
        except KeyboardInterrupt:
            self.save_checkpoint()
        save_config(self.log_name, 'Training done.')