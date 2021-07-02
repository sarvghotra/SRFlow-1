import os
import time
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
        self.hr_images = self.select_imgs(hr_file_path, selection_txt_file)
        self.lr_images = None

        gpu = True
        self.augment = opt["augment"] if "augment" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        t = time.time()


        t = time.time() - t
        print("{} Loaded {} HR images".format(self.phase, len(self.hr_images)))

        self.gpu = gpu
        self.rand_gauss_var = opt["gauss_noise_var"]
        print("rand gauss var: ", self.rand_gauss_var)

        self.measures = None
        self.i = 0

    def select_imgs(self, hr_dir_path, selection_txt_file):
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
