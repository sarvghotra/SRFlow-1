import os
import torch.utils.data as data
from data.LRHR_dataset import LRHRDataset

class LRHRSelectiveDataset(LRHRDataset):
    '''
    Selects only the images from the whole data that are mentioned in
    "selection file" to train on particular section of the data like
    faces etc.
    '''
    def __init__(self, opt):
        super(LRHRSelectiveDataset, self).__init__(opt)

        selection_txt_file = opt['selection_file']
        print("Pre size: ", len(self.hr_images))
        self.hr_images = self.select_imgs(selection_txt_file)
        print("Post size: ", len(self.hr_images))

    def select_imgs(self, selection_txt_file):
        '''
        Filters out images that are not given in the selection_txt_file
        '''
        selected_filenames = []
        # read the filenames
        with open(selection_txt_file, 'r', encoding='utf-8') as fi:
            for line in fi:
                line = line.strip()
                filename = line.split('\t')[0]
                selected_filenames.append(filename)

        selected_filenames = set(selected_filenames)
        selected_hr_imgs = []

        for img_path in self.hr_images:
            filename = os.path.basename(img_path)
            if filename in selected_filenames:
                selected_hr_imgs.append(img_path)

        return selected_hr_imgs
