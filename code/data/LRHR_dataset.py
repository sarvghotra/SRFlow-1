import imageio
import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch
import random
import skimage
import math
import lmdb
import pyarrow as pa
from PIL import Image
import six
import cv2

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

        gpu = True
        self.augment = opt["augment"] if "augment" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        t = time.time()

        self.lr_images = None
        if lr_file_path is not None:
            self.lr_images = utils.get_paths_from_images(lr_file_path)
            print("{} Loaded {} LR images".format(self.phase, len(self.lr_images)))

        self.hr_env = None
        if opt['env'] != 'lmdb':
            self.hr_images = utils.get_paths_from_images(hr_file_path)
            self.length = len(self.hr_images)
            print("{} Loaded {} HR images".format(self.phase, len(self.hr_images)))
        else:
            idx_file = os.path.join(os.path.split(hr_file_path)[0], 'train_images_idx.txt')
            self.length = 0
            with open(idx_file, 'r') as fi:
                for _ in fi:
                    self.length += 1

        #min_val_hr = np.min([i.min() for i in self.hr_images[:20]])
        #max_val_hr = np.max([i.max() for i in self.hr_images[:20]])

        #min_val_lr = np.min([i.min() for i in self.lr_images[:20]])
        #max_val_lr = np.max([i.max() for i in self.lr_images[:20]])

        t = time.time() - t
        #print("Loaded {} HR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
        #      format(len(self.hr_images), min_val_hr, max_val_hr, t, hr_file_path))
        #print("Loaded {} LR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
        #      format(len(self.lr_images), min_val_lr, max_val_lr, t, lr_file_path))

        self.gpu = gpu
        self.rand_gauss_var = opt["gauss_noise_var"]
        print("rand gauss var: ", self.rand_gauss_var)

        self.measures = None
        self.i = 0
        self.t_read = []
        self.imgs_path = []
        self.t_proc = []
        self.times = []

    def __len__(self):
        return self.length

    def _lr_img_from_hr(self, hr):
        # Bicubic downsample hr
        h, w = hr.shape[:2]
        rescale_h = h // self.scale
        rescale_w = w // self.scale
        lr = utils.imresize(hr, output_shape=(rescale_h, rescale_w))
        return lr

    def loads_pyarrow(self, buf):
        """
        Args:
            buf: the output of `dumps`.
        """
        return pa.deserialize(buf)

    def _read_hr_img(self, item):
        # Reads image(s) from storage
        if self.hr_env is not None:
            with self.hr_env.begin(write=False) as txn:
                byteflow = txn.get(self.keys[item])

            unpacked = self.loads_pyarrow(byteflow)

            imgbuf = unpacked
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            #hr = Image.open(buf).convert('RGB')
            #hr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            #hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            hr = imageio.imread(buf)
            if len(hr.shape) == 2:
                hr = np.stack((hr,) * 3, axis=-1)
                hr = hr[:, :, :3]
            hr_path = str(self.keys[item])
        else:
            hr_path = self.hr_images[item]
            hr = utils.img_loader(hr_path, lib='cv')
        return hr, hr_path

    def _get_hr_lr_img(self, item, tries=0):
        hr, hr_path = self._read_hr_img(item)
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

    def _init_lmdb(self):
        hr_file_path = self.opt['dataroot_GT']
        self.hr_env = lmdb.open(hr_file_path, subdir=os.path.isdir(hr_file_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.hr_env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            # self.length = self.loads_pyarrow(txn.get(b'__len__'))
            # self.keys = umsgpack.unpackb(txn.get(b'__keys__'))
            self.keys = self.loads_pyarrow(txn.get(b'__keys__'))

    def __getitem__(self, item):
        if self.opt['env'] == 'lmdb' and self.hr_env is None:
            self._init_lmdb()
        #st = time.time()
        hr, lr, path = self._get_hr_lr_img(item)
        #read_time = time.time() - st
        #self.t_read.append(time.time() - st)

        proc_st = time.time()

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


        #t_proc = time.time() - proc_st

        self.i += 1
        if self.i % 100 == 0 and False:
            import statistics
            print("Read Median: ", statistics.median(self.t_read))
            print("Read Avg: ", sum(self.t_read)/len(self.t_read))
            print("Read Highest: ", max(self.t_read))

            print("Proc Median: ", statistics.median(self.t_proc))
            print("Proc Avg: ", sum(self.t_proc)/len(self.t_proc))
            print("Proc Highest: ", max(self.t_proc))
            print()

        '''
        worker_id = torch.utils.data.get_worker_info().id

        # FIXME: remove it, temp
        # sanity check
        hr_img = self.rgb(hr)
        lr_img = self.rgb(lr)

        save_path = "/tmp/imgs/" + str(item) + "_lr.png"
        utils.save_img(lr_img, save_path)
        save_path = "/tmp/imgs/" + str(item) + "_hr.png"
        utils.save_img(hr_img, save_path)
        print("Saved")

        #total_time = time.time() - st
        #self.times.append(total_time)

        #if self.i % 8 == 0:
        #    print("batch time: ", sum(self.times))
        #    self.times = []
        '''

        return {'LQ': lr, 'GT': hr, 'LQ_path': path, 'GT_path': path} #, 'read_times': [read_time], 'worker_id': [worker_id], 'proc_times': [t_proc]}

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
