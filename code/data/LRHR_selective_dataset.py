import os
import time
import numpy as np
from data import utils
import random

import torch.utils.data as data
from data.LRHR_dataset import LRHRDataset

class LRHRSelectiveDataset(LRHRDataset):
    '''
    Selects only the images from the whole data that are mentioned in
    "selection file" to train on particular section of the data like
    faces etc.
    '''
    def __init__(self, opt):
        self.opt = opt
        self.phase = opt["name"]
        self.crop_size = opt.get("GT_size", None)
        self.scale = opt.get("scale")
        self.random_scale_list = [1]

        selection_txt_file = opt['selection_file']
        hr_file_path = opt["dataroot_GT"]
        self.hr_images = self._select_imgs(hr_file_path, selection_txt_file)
        self.length = len(self.hr_images)
        self.lr_images = None

        self.hr_env = None
        if opt['env'] == 'lmdb':
            self.img_to_lmdb_index = self._build_img_to_lmdb_index()

        gpu = True
        self.augment = opt["augment"] if "augment" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        t = time.time()


        t = time.time() - t
        print("{} Loaded {} HR images".format(self.phase, len(self.length)))

        self.gpu = gpu
        self.rand_gauss_var = opt["gauss_noise_var"]
        print("rand gauss var: ", self.rand_gauss_var)

        self.measures = None
        self.i = 0

    def _select_imgs(self, hr_dir_path, selection_txt_file):
        '''
        Filters out images that are not given in the selection_txt_file
        '''
        selected_files = []
        # read the filenames
        with open(selection_txt_file, 'r', encoding='utf-8') as fi:
            for line in fi:
                line = line.strip()
                filename = line.split('\t')[0]
                selected_files.append(os.path.join(hr_dir_path, filename))

        return selected_files

    def _build_img_to_lmdb_index(self):
        lmdb_dir = os.path.split(self.opt['dataroot_GT'])[0]
        img_to_lmdb_index_file = os.path.join(lmdb_dir, 'train_images_idx.txt')
        img_to_lmdb_index = {}
        with open(img_to_lmdb_index_file, 'r', encoding='utf-8') as fi:
            for line in fi:
                line = line.strip()
                img_name, index = line.split(' ')
                img_to_lmdb_index[img_name] = int(index)
        return img_to_lmdb_index

    def _get_hr_lr_img(self, item, tries=0):
        tries += 1
        if tries > 3:
            # if 3 ties fail
            # return a safe hardcoded img
            item = 0

        if self.hr_env is not None:
            img_name = os.split(self.hr_images[item])[1]
            item = self.img_to_lmdb_index(img_name)

        hr, hr_path = self._read_hr_img(item)

        # make hr shape multiple of scale
        h, w = hr.shape[:2]
        rescale_h = h // (self.scale * 2)
        rescale_w = w // (self.scale * 2)

        # hr_prime
        if self.opt.get('two_level_downsample', False):
            print("TWO level downsample: ", self.opt.get('two_level_downsample', False))
            hr = self._lr_img_from_hr(hr)

        # mod crop
        # extra 2 because LR needs to be even
        hr = hr[:rescale_h * self.scale * 2, :rescale_w * self.scale * 2, :]

        # hr image becomes too small to crop a crop_size patch from it
        h, w = hr.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            print("*************** Samll HR Image *********************: ", tries)
            item = random.randint(0, self.__len__() - 1)
            return self._get_hr_lr_img(item)

        # Check if LR is already given
        if self.lr_images:
            lr_path = self.lr_images[item]
            lr = utils.img_loader(lr_path)
            return hr, lr, ""

        # Create LR image on the fly from HR
        lr = self._lr_img_from_hr(hr)
        return hr, lr, hr_path
