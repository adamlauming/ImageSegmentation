'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2020-12-09 17:22:39
'''
import os
import sys
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import datasets.utils_data as utils


class DatasetAMDSEG(Dataset):
    def __init__(self, data_dir, Flags, mode='train'):
        super().__init__()
        self.mode = mode
        self.Flags = Flags
        self.inputsize = [512, 512]
        self.types = Flags.types


        self.image_dir = os.path.join(data_dir, 'Images')
        self.label_dir = os.path.join(data_dir, 'LesionLabels')
        self.listfile = os.path.join(data_dir, '{}files.txt'.format(mode))

        self.filenames = utils.txt2list(self.listfile)
        print("Num of {} images:  {}".format(mode, len(self.filenames)))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        batch_name = self.filenames[index]  # only 1 image name return
        image_arr, label_arr = self.get_img(batch_name, mode=self.mode)

        x_data = self.to_tensor(image_arr.copy()).float()
        y_data = self.to_tensor(label_arr.copy()).float()

        return x_data, y_data, batch_name

    # load images and labels depend on filenames
    def get_img(self, file_name, mode='train'):
        if 'train' in self.mode:
            image_file = os.path.join(self.image_dir, "{}.jpg".format(file_name))
            image_im = Image.open(image_file).resize(self.inputsize)
            label_im = self.LoadLabels(file_name)

            image_im = utils.random_perturbation(image_im)
            image_im, label_im = utils.random_geometric2(image_im, label_im)

            image_arr = np.array(image_im, dtype=np.float32) / 255.0
            label_arr = [np.array(im, dtype=np.float32) / 255.0 for im in label_im]
            label_arr = np.dstack(label_arr)
            label_arr = (label_arr > 0.5) * 1.0

            return image_arr, label_arr

        elif 'val' in self.mode:
            image_file = os.path.join(self.image_dir, "{}.jpg".format(file_name))
            image_im = Image.open(image_file).resize(self.inputsize)
            label_im = self.LoadLabels(file_name)

            image_arr = np.array(image_im, dtype=np.float32) / 255.0
            label_arr = [np.array(im, dtype=np.float32) / 255.0 for im in label_im]
            label_arr = np.dstack(label_arr)
            label_arr = (label_arr > 0.5) * 1.0

            return image_arr, label_arr

        elif 'test' in self.mode:
            image_file = os.path.join(self.image_dir, "{}.jpg".format(file_name))
            image_im = Image.open(image_file).resize(self.inputsize)

            image_arr = np.array(image_im, dtype=np.float32) / 255.0

            return image_arr, image_arr


    def LoadLabels(self, filename):
        # types = ['disc', 'drusen', 'exudate', 'hemorrhage', 'scar', 'others']

        labels = []
        for tag in self.types:
            img = Image.open(os.path.join(self.label_dir, filename + '_' + tag + '.bmp'))
            labels.append(img.resize(self.inputsize))

        return labels



