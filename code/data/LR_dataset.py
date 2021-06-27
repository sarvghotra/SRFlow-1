import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch
import random


from data import utils


class LRDataset(data.Dataset):
    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.random_scale_list = [1]

        lr_file_path = opt["dataroot_LQ"]
        y_labels_file_path = opt['dataroot_y_labels']

        gpu = True
        t = time.time()
        self.lr_images = utils.get_paths_from_images(lr_file_path)

        t = time.time() - t
        print("Loaded {} LR images".format(len(self.lr_images)))

        self.gpu = gpu
        self.measures = None

    def __len__(self):
        return len(self.lr_images)

    def to_tensor(self, array):
        return torch.Tensor(array)

    # convert to image
    def rgb(self, t):
        return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)

    def __getitem__(self, item):
        lr_path = self.lr_images[item]
        lr = utils.img_loader(lr_path)

        # Pad image to be % 2
        pad_factor = 2
        h, w, c = lr.shape
        lr = utils.impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                right=int(np.ceil(w / pad_factor) * pad_factor - w))

        lr = lr / 255.0
        # HWC to CHW
        lr = np.ascontiguousarray(lr.transpose([2, 0, 1])).astype(np.float32)
        lr = self.to_tensor(lr)   #torch.Tensor(lr)

        return {'GT': lr, 'GT_path':lr_path, 'LQ': lr, 'LQ_path': lr_path}
