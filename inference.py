#%%
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from datasets.DataAMD import *
from models.Losses import *
from models.choose_model import seg_model
from utils import utils
from utils.logger import Logger
from utils.Metrics import *
import pandas as pd

from skimage.morphology import remove_small_holes, remove_small_objects

utils.log('start inference')

#%%
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='1', type=str, help="get center type")
parser.add_argument('--mode', default='val', type=str, help="get center type")
parser.add_argument('--types', default=['drusen', 'exudate', 'hemorrhage'], nargs='+', help="output lensions")
Flags, _ = parser.parse_known_args()

# load model
pypath = os.path.abspath(__file__)
path, _ = os.path.split(pypath)
if not Flags.model:
    model_dir = os.path.join(path, '..', 'T1210-1452-Unet-Res18', 'Model')
    weightnames = utils.all_files_under(model_dir, extension='pkl', append_path=True)
    print(weightnames)
    weightname = weightnames[-1]
    model = torch.load(weightname)
else:
    weightname = os.path.join(path, '..', 'T1211-1059-AttUnet', 'Model', 'Model_Ep150_Dice_0.8525.pkl')
    model = torch.load(weightname)
model.eval()

# load data
data_dir = os.path.join(path, 'datasets', 'TestImages')
filenames = utils.all_files_under(data_dir, extension='jpg', append_path=False)
types = Flags.types

# save config
save_dir = os.path.join(path, '..', 'Lesion_Segmentation')
lesions = ['Drusen', 'Exudate', 'Hemorrhage', 'Others', 'Scar']
lesionmap = {
    'drusen': 'Drusen',
    'exudate': 'Exudate',
    'hemorrhage': 'Hemorrhage',
    'others': 'Others',
    'scar': 'Scar',
}
for lesion in lesions:
    lension_dir = os.path.join(save_dir, lesion)
    utils.checkpath(lension_dir)

#%% Evaluation
to_tensor = transforms.ToTensor()
with torch.no_grad():
    for idx, filename in enumerate(filenames):
        print(filename)
        image = Image.open(os.path.join(data_dir, filename))
        rawsize = image.size
        image = image.resize([512, 512])
        image_arr = np.array(image, dtype=np.float32) / 255.0

        x_data = to_tensor(image_arr.copy()).float()
        if torch.cuda.is_available():
            x_data = x_data.cuda()
        x_data = torch.unsqueeze(x_data, dim=0)

        y_pred = model(x_data)["out"]
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.cpu().data.numpy()
        y_pred = np.squeeze(y_pred)

        name, _ = os.path.splitext(filename)
        for i in range(len(types)):
            y_pred_ = y_pred[i, :, :]
            y_pred_ = (y_pred_ > 0.4)
            y_pred_ = remove_small_objects(y_pred_, min_size=32)
            y_pred_ = remove_small_holes(y_pred_, area_threshold=64)
            y_pred_ = 1 - y_pred_ * 1.0

            im = utils.array2image(y_pred_)
            im = im.resize(rawsize)
            im.save(os.path.join(save_dir, lesionmap[types[i]], name+'.png'))


