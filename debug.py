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
from models.network import *
from utils import utils
from utils.logger import Logger
from utils.Metrics import *

#%% Settings
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_rate', default=0.5, type=float, help="gpu free rate")
parser.add_argument('--gpu_gap', default=10, type=int, help="check free gpu interval")
parser.add_argument('--log_cols', default=140, type=int, help="num of columns for log")

parser.add_argument('--epochs', default=100, type=int, help="nums of epoch")
parser.add_argument('--types', default=['drusen', 'other'], nargs='+', help="output lensions")
parser.add_argument('--datatrain', default='trainDemo', type=str, help="select all data or part data train")
parser.add_argument('--batch_size', default=1, type=int, help="batch size")
parser.add_argument('--workers', default=0, type=int, help="num of workers")

parser.add_argument('--model', default='PANet', type=str, help="training model")
parser.add_argument('--en_pretrained', default=False, type=bool, help="whether load pretrained model")
parser.add_argument('--learning_rate', default=3e-4, type=float, help="initial learning rate for Adam")
parser.add_argument('--alpha', default=2, type=float, help="focal loss weigth")
parser.add_argument('--val_step', default=5, type=int, help="the frequency of evaluation")

parser.add_argument('--savename', default='Result', type=str, help="output folder name")

Flags, _ = parser.parse_known_args()
utils.ShowFlags(Flags)
os.environ['CUDA_VISIBLE_DEVICES'] = utils.SearchFreeGPU(Flags.gpu_gap, Flags.gpu_rate)

#==============================================================================
# Dataset
#==============================================================================
data_dir = os.path.join('..', '..', 'Dataset')
dataset_train = DatasetAMDSEG(data_dir, Flags, mode=Flags.datatrain)
dataloader_train = DataLoader(dataset_train, batch_size=Flags.batch_size, num_workers=Flags.workers, shuffle=True)

dataset_val = DatasetAMDSEG(data_dir, Flags, mode=Flags.datatrain.replace('train', 'val'))
dataloader_val = DataLoader(dataset_val, batch_size=Flags.batch_size, num_workers=Flags.workers, shuffle=True)

channels = len(Flags.types)
utils.log('Load Data successfully')

# #==============================================================================
# # Logger
# #==============================================================================
# logger = Logger(Flags)
# utils.log('Setup Logger successfully')

#==============================================================================
# load model, optimizer, Losses
#==============================================================================
model = globals()[Flags.model](out_channels=channels, pretrained=Flags.en_pretrained)
print('load model {}'.format(Flags.model))

# summary(model, (3, 512, 512))
model = model.cuda() if torch.cuda.is_available() else model
optimizer = torch.optim.Adam(model.parameters(), lr=Flags.learning_rate)
criterion = {
    'BCE': BCELoss(),
    'DICE': DiceLoss(),
    'FOCAL': FocalLoss(gamma=2, alpha=[Flags.alpha, 1]),
}
metrics = AMDSegMetrics()
utils.log('Build Model successfully')
#%% Training
# for epoch in range(Flags.epochs):

############################################################
# Train Period
############################################################
epoch = 0
model.train()
# pbar = tqdm(dataloader_train, ncols=Flags.log_cols)
# pbar.set_description('Epoch {:2d}'.format(epoch))

log_Train, temp = {}, {}

for n_step, batch_data in enumerate(dataloader_train):
    # get data
    x_batch = batch_data[0]
    y_batch = batch_data[1]


    if torch.cuda.is_available():
        x_data = x_batch.cuda()
        y_true = y_batch.cuda()
    optimizer.zero_grad()

    # forward
    y_pred = model(x_data)
    print(y_pred.shape)
    break

    loss = criterion['PCALoss'](feature_de[0])
    print(loss)
    loss.backward()
    optimizer.step()


    y_true = y_true.cpu().data.numpy()
    y_pred = y_pred.cpu().data.numpy()
    if n_step == 0:
        break
