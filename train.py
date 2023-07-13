'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2020-12-09 17:22:39
'''
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


#%% Settings
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_rate', default=0.5, type=float, help="gpu free rate")
parser.add_argument('--gpu_gap', default=10, type=int, help="check free gpu interval")
parser.add_argument('--log_cols', default=140, type=int, help="num of columns for log")

parser.add_argument('--epochs', default=150, type=int, help="nums of epoch")
parser.add_argument('--inchannels', default=3, type=int, help="nums of input channels")
parser.add_argument('--types', default=['drusen', 'exudate', 'hemorrhage'], nargs='+', help="output lensions")
parser.add_argument('--datatrain', default='train', type=str, help="select all data or part data train")
parser.add_argument('--batch_size', default=4, type=int, help="batch size")
parser.add_argument('--workers', default=2, type=int, help="num of workers")

parser.add_argument('--model', default='PSPNetSMP', type=str, help="training model")
parser.add_argument('--encoder', default='resnet18', type=str, help="training model encoder")
parser.add_argument('--en_pretrained', default=True, type=bool, help="whether load pretrained model")
parser.add_argument('--dilated', default=False, type=bool, help="whether dilated")
parser.add_argument('--deep_stem', default=False, type=bool, help="whether deep_stem")
parser.add_argument('--aux', default=False, type=bool, help="whether aux")
parser.add_argument('--learning_rate', default=2e-4, type=float, help="initial learning rate for Adam")
parser.add_argument('--alpha', default=2, type=float, help="focal loss weigth")
parser.add_argument('--val_step', default=5, type=int, help="the frequency of evaluation")


parser.add_argument('--savename', default='Result', type=str, help="output folder name")

Flags, _ = parser.parse_known_args()
utils.ShowFlags(Flags)
os.environ['CUDA_VISIBLE_DEVICES'] = utils.SearchFreeGPU(Flags.gpu_gap, Flags.gpu_rate)
torch.cuda.empty_cache()

#==============================================================================
# Dataset
#==============================================================================
data_dir = os.path.join('..', 'codes', 'datasets')
dataset_train = DatasetAMDSEG(data_dir, Flags, mode=Flags.datatrain)
dataloader_train = DataLoader(dataset_train, batch_size=Flags.batch_size, num_workers=Flags.workers, shuffle=True)

dataset_val = DatasetAMDSEG(data_dir, Flags, mode=Flags.datatrain.replace('train', 'val'))
dataloader_val = DataLoader(dataset_val, batch_size=Flags.batch_size, num_workers=Flags.workers, shuffle=True)

classes = len(Flags.types)
utils.log('Load Data successfully')

#==============================================================================
# Logger
#==============================================================================
logger = Logger(Flags)
utils.log('Setup Logger successfully')

#==============================================================================
# load model, optimizer, Losses
#==============================================================================
# model = globals()[Flags.model](Flags)
model = seg_model(Flags)
model = model.cuda() if torch.cuda.is_available() else model
# summary(model, (3, 512, 512))
optimizer = torch.optim.Adam(model.parameters(), lr=Flags.learning_rate)
criterion = {
    'BCE': BCELoss(),
    'DICE': DiceLoss(),
    'FOCAL': FocalLoss(gamma=2, alpha=[Flags.alpha, 1]),
}
metrics = AMDSegMetrics()
utils.log('Build Model successfully')
#==============================================================================
# Train model
#==============================================================================
for epoch in range(Flags.epochs + 1):
    ############################################################
    # Train Period
    ############################################################
    model.train()
    pbar = tqdm(dataloader_train, ncols=Flags.log_cols)
    pbar.set_description('Epoch {:2d}'.format(epoch))

    log_Train, temp = {}, {}
    for n_step, batch_data in enumerate(pbar):
        # get data
        x_batch = batch_data[0]
        y_batch = batch_data[1]

        if torch.cuda.is_available():
            x_data = x_batch.cuda()
            y_true = y_batch.cuda()
        optimizer.zero_grad()
        # forward
        y_pred = model(x_data)["out"]
        y_pred = torch.sigmoid(y_pred)

        # backward
        loss_bce = criterion['FOCAL'](y_true, y_pred)
        loss_dice = criterion['DICE'](y_true, y_pred)
        loss = loss_bce + loss_dice
        loss.backward()
        optimizer.step()

        # log
        temp['Loss'] = loss.item()
        log_Train = utils.MergeLog(log_Train, temp, n_step)
        pbar.set_postfix(log_Train)
    logger.write_tensorboard('1.Train', log_Train, epoch)

    if not (epoch % Flags.val_step == 0):
        continue

    ############################################################
    # Test Period
    ############################################################
    print('*' * Flags.log_cols)
    with torch.no_grad():
        model.eval()
        pbar = tqdm(dataloader_val, ncols=Flags.log_cols)
        pbar.set_description('Val')
        log_Test, temp = {}, {}
        for n_step, batch_data in enumerate(pbar):
            # get data
            x_batch = batch_data[0]
            y_batch = batch_data[1]
            if torch.cuda.is_available():
                x_data = x_batch.cuda()
                y_true = y_batch.cuda()

            # forward
            y_pred = model(x_data)["out"]
            y_pred = torch.sigmoid(y_pred)
            loss_bce = criterion['FOCAL'](y_true, y_pred)
            loss_dice = criterion['DICE'](y_true, y_pred)
            loss = loss_bce + loss_dice
            temp['Loss'] = loss.item()

            # metric
            y_true = y_true.cpu().data.numpy()
            y_pred = y_pred.cpu().data.numpy()
            dice = metrics.SegmentationMetrics(y_true, y_pred, 'middle')
            # _, _, _, _, f1 = metrics.ClassifyMetric(y_true, y_pred, 'middle')
            # _, roc = metrics.AUC(y_true, y_pred)

            types = Flags.types
            for i in range(len(types)):
                temp['Dice_' + types[i]] = dice[i]

            log_Test = utils.MergeLog(log_Test, temp, n_step)
            pbar.set_postfix(log_Test)
        logger.write_tensorboard('2.Val', log_Test, epoch)
        logger.save_model(model, 'Ep{}_Dice_{:.4f}'.format(epoch, log_Test['Dice_drusen']))
    print('*' * Flags.log_cols)
