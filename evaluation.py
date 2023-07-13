#%%
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
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

utils.log('start evaluation')


def DiceScore(x, y, eps=1e-4):

    return (np.sum(x * y) + eps) / (np.sum(x) + np.sum(y) + eps)


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
    model_dir = os.path.join(path, '..', 'T1210-1034-PSPNetSMP-Res18', 'Model')
    weightnames = utils.all_files_under(model_dir, extension='pkl', append_path=True)
    print(weightnames)
    weightname = weightnames[-1]
    model = torch.load(weightname)
else:
    weightname = os.path.join(path, '..', 'T1213-1156-PANNetSMP-Res18', 'Model', 'Model_Ep140_Dice_0.9245.pkl')
    model = torch.load(weightname)
model.eval()

# load data
data_dir = os.path.join(path, 'datasets')
dataset_val = DatasetAMDSEG(data_dir, Flags, mode=Flags.mode)
dataloader_val = DataLoader(dataset_val, batch_size=4)
types = Flags.types

# save config
_, name = os.path.split(weightname)
name, _ = os.path.splitext(name)
save_dir = os.path.join(path, '..', 'ModelResults', 'PANNetSMP_'+name)
print(save_dir)
utils.checkpath(save_dir)

#%% Evaluation
pbar = tqdm(dataloader_val, ncols=60)
df_Dice = pd.DataFrame(columns=['Name'] + Flags.types)
df_Dice.set_index('Name')

with torch.no_grad():
    for n_step, batch_data in enumerate(pbar):
        # get data
        x_batch = batch_data[0]
        y_batch = batch_data[1]
        data_name = batch_data[2]
        if torch.cuda.is_available():
            x_data = x_batch.cuda()
            y_true = y_batch.cuda()

        y_pred = model(x_data)["out"]
        y_pred = torch.sigmoid(y_pred)

        y_true = y_true.cpu().data.numpy()
        y_pred = y_pred.cpu().data.numpy()

        for batchidx in range(x_data.shape[0]):
            name_ = data_name[batchidx]
            for i in range(len(types)):
                y_pred_ = y_pred[batchidx, i, :, :]
                y_pred_ = (y_pred_ > 0.4)
                y_pred_ = remove_small_objects(y_pred_, min_size=32)
                y_pred_ = remove_small_holes(y_pred_, area_threshold=64)
                y_pred_ = y_pred_ * 1.0

                utils.array2image(y_pred_).save(os.path.join(save_dir, '{}_{}_pred.png'.format(name_, types[i])))
                y_true_ = y_true[batchidx, i, :, :]
                utils.array2image(y_true_).save(os.path.join(save_dir, '{}_{}_true.png'.format(name_, types[i])))

                df_Dice.loc[name_, types[i]] = DiceScore(y_pred_, y_true_)

df_Dice.to_csv(os.path.join(save_dir, 'Dice.csv'))
print(df_Dice.mean())