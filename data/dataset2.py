#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-23 14:57:22
LastEditTime: 2021-08-21 00:13:37
@Description: file content
'''
import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F
# import rasterio
from scipy import io as sio


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF','.mat'])


def load_img(filepath):
    img = Image.open(filepath)
    return img


def load_MSmat(file_path):
    # print(file_path)
    data = sio.loadmat(file_path)  # HxWxC
    # print("load_dataset_singlemat: ", data.keys())
    # tensor type:
    img = data['imgMS']

    # print("min:{}, max:{}".format(min_v,max_v))
    imgn = img / 2047.0
    # max_v = img.max()
    # min_v = img.min()
    # imgn = (img - min_v) / (max_v - min_v)
    img8 = (imgn * 255).astype(np.uint8)
    # print(file_path)


    imgp = Image.fromarray(img8)
    # max_v = imgp.max()
    # min_v = imgp.min()
    # print("min:{}, max:{}".format(min_v,max_v))
    return imgp

def load_PANmat(file_path):
    data = sio.loadmat(file_path)  # HxWxC
    # print("load_dataset_singlemat: ", data.keys())
    # tensor type:
    img = data['imgPAN']
    imgn = img / 2047.0
    # max_v = img.max()
    # min_v = img.min()
    # imgn = (img - min_v) / (max_v - min_v)
    img8 = (imgn * 255).astype(np.uint8)
    imgp = Image.fromarray(img8)
    return imgp


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def rescale_img_NEAR(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.NEAREST)
    return img_in


def get_patch(ms_image, lms_image, pan_image, bms_image, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = lms_image.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    lms_image = lms_image.crop((iy, ix, iy + ip, ix + ip))
    ms_image = ms_image.crop((ty, tx, ty + tp, tx + tp))
    pan_image = pan_image.crop((ty, tx, ty + tp, tx + tp))
    bms_image = bms_image.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return ms_image, lms_image, pan_image, bms_image, info_patch


def augment(ms_image, lms_image, pan_image, bms_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        ms_image = ImageOps.flip(ms_image)
        lms_image = ImageOps.flip(lms_image)
        pan_image = ImageOps.flip(pan_image)
        bms_image = ImageOps.flip(bms_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            ms_image = ImageOps.mirror(ms_image)
            lms_image = ImageOps.mirror(lms_image)
            pan_image = ImageOps.mirror(pan_image)
            bms_image = ImageOps.mirror(bms_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            ms_image = ms_image.rotate(180)
            lms_image = lms_image.rotate(180)
            pan_image = pan_image.rotate(180)
            bms_image = pan_image.rotate(180)
            info_aug['trans'] = True

    return ms_image, lms_image, pan_image, bms_image, info_aug


class Data(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, data_dir_mtf, cfg, transform=None):
        super(Data, self).__init__()

        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.downgrade = cfg['data']['downgrade']
        if self.downgrade == 'MTF':
            self.mtf_image_filenames = [join(data_dir_mtf, x) for x in listdir(data_dir_mtf) if is_image_file(x)]

        self.cfg = cfg

    def __getitem__(self, index):

        ms_image = load_MSmat(self.ms_image_filenames[index])
        pan_image = load_PANmat(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])

        if self.downgrade == 'Bic':
            #1024
            pan_image = pan_image.resize(
                (int(pan_image.size[0] / self.upscale_factor), int(pan_image.size[1] / self.upscale_factor)),
                Image.BICUBIC)
            # 256
            pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                        pan_image.size[1] // self.upscale_factor * self.upscale_factor))
            ms_image = ms_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                      pan_image.size[1] // self.upscale_factor * self.upscale_factor))
            lms_image = ms_image.resize(
                (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)),
                Image.BICUBIC)
            bms_image = rescale_img(lms_image, self.upscale_factor)



        elif self.downgrade == 'MTF':
            lms_image = load_img(self.mtf_image_filenames[index])
            ms_image = ms_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                      pan_image.size[1] // self.upscale_factor * self.upscale_factor))
            lms_image = ms_image.resize(
                (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)),
                Image.NEAREST)
            bms_image = rescale_img_NEAR(lms_image, self.upscale_factor)
            #1024
            pan_image = pan_image.resize(
                (int(pan_image.size[0] / self.upscale_factor), int(pan_image.size[1] / self.upscale_factor)),
                Image.BICUBIC)
            # 256
            pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                        pan_image.size[1] // self.upscale_factor * self.upscale_factor))

        ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image,
                                                                 self.patch_size, scale=self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)

        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)


class Data_test(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, data_dir_mtf, cfg, transform=None):
        super(Data_test, self).__init__()

        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.downgrade = cfg['data']['downgrade']
        if self.downgrade == 'MTF':
            self.mtf_image_filenames = [join(data_dir_mtf, x) for x in listdir(data_dir_mtf) if is_image_file(x)]

        self.cfg = cfg

    def __getitem__(self, index):

        ms_image = load_MSmat(self.ms_image_filenames[index])
        pan_image = load_PANmat(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        # testfull = True
        # if testfull:
        #     bms_image = rescale_img(ms_image, self.upscale_factor)
        #     lms_image = ms_image
        #     ms_image = bms_image

        # else:
        if self.downgrade == 'Bic':
                # 1024
                pan_image = pan_image.resize(
                    (int(pan_image.size[0] / self.upscale_factor), int(pan_image.size[1] / self.upscale_factor)),
                    Image.BICUBIC)
                # 256
                pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                            pan_image.size[1] // self.upscale_factor * self.upscale_factor))

                ms_image = ms_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                          pan_image.size[1] // self.upscale_factor * self.upscale_factor))
                lms_image = ms_image.resize(
                    (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)),
                    Image.BICUBIC)
                bms_image = rescale_img(lms_image, self.upscale_factor)

        elif self.downgrade == 'MTF':
                lms_image = load_img(self.mtf_image_filenames[index])
                ms_image = ms_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                          pan_image.size[1] // self.upscale_factor * self.upscale_factor))
                lms_image = ms_image.resize(
                    (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)),
                    Image.NEAREST)
                bms_image = rescale_img_NEAR(lms_image, self.upscale_factor)
                # 1024
                pan_image = pan_image.resize(
                    (int(pan_image.size[0] / self.upscale_factor), int(pan_image.size[1] / self.upscale_factor)),
                    Image.BICUBIC)
                # 256
                pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                            pan_image.size[1] // self.upscale_factor * self.upscale_factor))

            # pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
            #                             pan_image.size[1] // self.upscale_factor * self.upscale_factor))




        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)

        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)


class Data_eval(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, data_dir_mtf, cfg, transform=None):
        super(Data_eval, self).__init__()

        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):

        lms_image = load_MSmat(self.ms_image_filenames[index])
        pan_image = load_PANmat(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        bms_image = rescale_img(lms_image, self.upscale_factor)

        if self.transform:
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        return bms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)