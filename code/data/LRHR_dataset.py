import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch
import random
import skimage
import math

from data import utils
from data.LRHR_PKL_dataset import random_flip, \
    random_rotation, random_crop, center_crop


class LRHRDataset(data.Dataset):
    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.phase = opt["name"]
        self.crop_size = opt.get("GT_size", None)
        self.scale = opt.get("scale")
        self.random_scale_list = [1]

        hr_file_path = opt["dataroot_GT"]
        lr_file_path = opt["dataroot_LQ"]
        y_labels_file_path = opt['dataroot_y_labels']

        gpu = True
        self.augment = opt["augment"] if "augment" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        t = time.time()

        self.lr_images = None
        if lr_file_path is not None:
            self.lr_images = utils.get_paths_from_images(lr_file_path)
            print("{} Loaded {} LR images".format(self.phase, len(self.lr_images)))

        self.hr_images = utils.get_paths_from_images(hr_file_path)

        #min_val_hr = np.min([i.min() for i in self.hr_images[:20]])
        #max_val_hr = np.max([i.max() for i in self.hr_images[:20]])

        #min_val_lr = np.min([i.min() for i in self.lr_images[:20]])
        #max_val_lr = np.max([i.max() for i in self.lr_images[:20]])

        t = time.time() - t
        #print("Loaded {} HR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
        #      format(len(self.hr_images), min_val_hr, max_val_hr, t, hr_file_path))
        #print("Loaded {} LR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
        #      format(len(self.lr_images), min_val_lr, max_val_lr, t, lr_file_path))
        print("{} Loaded {} HR images".format(self.phase, len(self.hr_images)))

        self.gpu = gpu
        self.rand_gauss_var = opt["gauss_noise_var"]
        print("rand gauss var: ", self.rand_gauss_var)

        self.measures = None
        self.i = 0

    def __len__(self):
        return len(self.hr_images)

    def _lr_img_from_hr(self, hr):
        # Bicubic downsample hr
        h, w = hr.shape[:2]
        rescale_h = h // self.scale
        rescale_w = w // self.scale
        lr = utils.imresize(hr, output_shape=(rescale_h, rescale_w))
        return lr

    def _get_hr_lr_img(self, item):
        # Reads image(s) from storage
        hr_path = self.hr_images[item]
        hr = utils.img_loader(hr_path)

        # make hr shape multiple of scale
        h, w = hr.shape[:2]
        rescale_h = h // (self.scale * 2)
        rescale_w = w // (self.scale * 2)
        # extra 2 because LR needs to be even
        hr = hr[:rescale_h * self.scale * 2, :rescale_w * self.scale * 2, :]

        # Check if LR is already given
        if self.lr_images:
            lr_path = self.lr_images[item]
            lr = utils.img_loader(lr_path)
            return hr, lr, ""

        # Create LR image on the fly from HR
        lr = self._lr_img_from_hr(hr)
        return hr, lr, hr_path

    def to_tensor(self, array):
        return torch.Tensor(array)

    # convert to image
    def rgb(self, t):
        return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)

    def __getitem__(self, item):
        hr, lr, path = self._get_hr_lr_img(item)

        if self.scale == None:
            self.scale = hr.shape[1] // lr.shape[1]

        assert hr.shape[1] == self.scale * lr.shape[1], ('non-fractional ratio', lr.shape, hr.shape, self.phase)

        if self.phase == "train" and self.use_crop:
            hr, lr = random_crop(hr, lr, self.crop_size, self.scale)

        #if self.center_crop_hr_size:
        #    hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size // self.scale)

        #if self.phase != "train" and self.opt['save_imgs']:
        # Pad image to be % 2
        '''
        pad_factor = 2
        h, w, c = lr.shape
        lr = utils.impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                right=int(np.ceil(w / pad_factor) * pad_factor - w))
        '''

        if self.phase == "train" and self.augment:
            hr, lr = utils.augment([hr, lr])

        hr = hr / 255.0
        lr = lr / 255.0

        if self.measures is None or random.random() < 0.05:
            if self.measures is None:
                self.measures = {}
            self.measures['hr_means'] = np.mean(hr)
            self.measures['hr_stds'] = np.std(hr)
            self.measures['lr_means'] = np.mean(lr)
            self.measures['lr_stds'] = np.std(lr)

        #if self.rand_gauss_var is not None:
        if self.phase == "train":
            hr = self.add_rand_gauss_noise(hr)

        # HWC to CHW
        hr = np.ascontiguousarray(hr.transpose([2, 0, 1])).astype(np.float32)
        lr = np.ascontiguousarray(lr.transpose([2, 0, 1])).astype(np.float32)
        hr = self.to_tensor(hr)   #torch.Tensor(hr)
        lr = self.to_tensor(lr)   #torch.Tensor(lr)


        '''
        # FIXME: remove it, temp
        # sanity check
        hr_img = self.rgb(hr)
        lr_img = self.rgb(lr)

        save_path = "/tmp/imgs/" + str(item) + "_lr.png"
        utils.save_img(lr_img, save_path)
        save_path = "/tmp/imgs/" + str(item) + "_hr.png"
        utils.save_img(hr_img, save_path)
        print("Saved")
        '''

        return {'LQ': lr, 'GT': hr, 'LQ_path': path, 'GT_path': path}

    def print_and_reset(self, tag):
        m = self.measures
        kvs = []
        for k in sorted(m.keys()):
            kvs.append("{}={:.2f}".format(k, m[k]))
        print("[KPI] " + tag + ": " + ", ".join(kvs))
        self.measures = None

    def add_rand_gauss_noise(self, img):
        img = skimage.util.random_noise(img, mode='gaussian',
                                        var=self.rand_gauss_var,
                                        seed=random.randint(0, 50000)
                                        )
        img = np.float32(img)
        img = np.clip(img, 0, 1)
        return img


def random_crop(hr, lr, patch_size, scale):
    # def get_patch(self, lr, hr, patch_size, scale, is_same_size=False):
    ih, iw = lr.shape[:2]

    ip = patch_size
    ip = patch_size // scale

    iy = random.randrange(0, ih - ip + 1)
    ix = random.randrange(0, iw - ip + 1)

    ty, tx = iy, ix
    ty, tx = scale * iy, scale * ix

    lr = lr[iy:iy + ip, ix:ix + ip, :]
    hr = hr[ty:ty + patch_size, tx:tx + patch_size, :]

    return hr, lr
