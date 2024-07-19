# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 19:25:02 2024

@author: MSP
"""

import time
import einops
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
from os.path import join
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.data as data
import h5py
import random
from Train_model import Net
import argparse

# Argument parser for command line options
parser = argparse.ArgumentParser(description="PyTorch Light Field Hybrid SR")

# Training settings
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=5, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default=96, help="Training patch size")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--resume_epoch", type=int, default=0, help="Resume from checkpoint epoch")
parser.add_argument("--num_cp", type=int, default=1, help="Number of epochs for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epochs for saving loss figure")
parser.add_argument("--dataset", type=str, default="HCI", help="Dataset for training")
parser.add_argument("--dataset_path", type=str, default="HCI_train_data.h5")
parser.add_argument("--angular_out", type=int, default=7, help="Angular number of the dense light field")
parser.add_argument("--angular_in", type=int, default=2, help="Angular number of the sparse light field [AngIn x AngIn]")
opt = parser.parse_args(args=[])

class DatasetFromHdf5(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromHdf5, self).__init__()
        
        # Load data from HDF5 file
        hf = h5py.File(opt.dataset_path)
        self.LFI = hf.get('LFI')  # [N,ah,aw,h,w]
        self.LFI = self.LFI[:, :opt.angular_out, :opt.angular_out, :, :]
   
        self.psize = opt.patch_size
        self.ang_out = opt.angular_out
        self.ang_in = opt.angular_in

    def __getitem__(self, index):
        # Get one item
        lfi = self.LFI[index]  # [ah,aw,h,w]

        # Crop to patch
        H = lfi.shape[2]
        W = lfi.shape[3]

        x = random.randrange(0, H - self.psize)
        y = random.randrange(0, W - self.psize)
        lfi = lfi[:, :, x:x + self.psize, y:y + self.psize]
        
        # 4D augmentation
        # Flip
        if np.random.rand(1) > 0.5:
            lfi = np.flip(np.flip(lfi, 0), 2)
        if np.random.rand(1) > 0.5:
            lfi = np.flip(np.flip(lfi, 1), 3)
        # Rotate
        r_ang = np.random.randint(1, 5)
        lfi = np.rot90(lfi, r_ang, (2, 3))
        lfi = np.rot90(lfi, r_ang, (0, 1))
            
        # Get input index
        ind_all = np.arange(self.ang_out * self.ang_out).reshape(self.ang_out, self.ang_out)
        delt = (self.ang_out - 1) // (self.ang_in - 1)
        ind_source = ind_all[0:self.ang_out:delt, 0:self.ang_out:delt]
        ind_source = ind_source.reshape(-1)

        # Get input and label    
        lfi = lfi.reshape(-1, self.psize, self.psize)
        input = lfi[ind_source, :, :]

        # Reformat input to lenslet format
        NumView2 = 2
        a = 0
        LF2 = np.zeros((1, self.psize * NumView2, self.psize * NumView2))
        for i in range(NumView2):
            for j in range(NumView2):
                img = input[a, :, :]
                img = img[np.newaxis, :, :]
                LF2[:, i::NumView2, j::NumView2] = img
                a += 1
        lenslet_data = LF2

        H = self.psize
        W = self.psize
        allah = self.ang_in
        allaw = self.ang_in
        LFI = np.zeros((1, H, W, allah, allaw))
        eslf = lenslet_data
        for v in range(allah):
            for u in range(allah):
                sub = eslf[:, v::allah, u::allah]
                LFI[:, :, :, v, u] = sub[:, 0:H, 0:W]
        LFI = LFI.reshape(self.psize, self.psize, 2, 2)

        # Convert to tensor
        input = torch.from_numpy(input.astype(np.float32) / 255.0)
        label = torch.from_numpy(lfi.astype(np.float32) / 255.0)
        LFI = torch.from_numpy(LFI.astype(np.float32) / 255.0)

        return ind_source, input, label, LFI

    def __len__(self):
        return self.LFI.shape[0]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# Create directory for model checkpoints
opt.num_source = opt.angular_in * opt.angular_in
model_dir = 'model_{}_S{}'.format(opt.dataset, opt.num_source)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Data loader
print('===> Loading datasets')
train_set = DatasetFromHdf5(opt)
train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
print('Loaded {} LFIs from {}'.format(len(train_loader), opt.dataset_path))

# Build model
print("Building net")
model = Net(opt).to(device)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
losslogger = defaultdict(list)

# Optionally resume from a checkpoint
if opt.resume_epoch:
    resume_path = join(model_dir, 'model_epoch_{}.pth'.format(opt.resume_epoch))
    if os.path.isfile(resume_path):
        print("==> Loading checkpoint 'epoch{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        losslogger = checkpoint['losslogger']
    else:
        print("==> No model found at 'epoch{}'".format(opt.resume_epoch))

# Loss function
def reconstruction_loss(X, Y):
    # L1 Charbonnier loss
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + eps)
    loss = torch.sum(error) / torch.numel(error)
    return loss

# Training function
def train(epoch):
    model.train()
    loss_count = 0.0

    for k in range(10):
        for i, batch in enumerate(train_loader, 1):
            ind_source, input, label, LFI = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            pred_views, pred_lf = model(ind_source, input, LFI, opt)
            loss = reconstruction_loss(pred_lf, label)
            for i in range(pred_views.shape[2]):
                loss += reconstruction_loss(pred_views[:, :, i, :, :], label)
            loss_count += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    scheduler.step()
    losslogger['epoch'].append(epoch)
    losslogger['loss'].append(loss_count / len(train_loader))
    return loss_count / len(train_loader)

# Training loop
min=10
for epoch in range(opt.resume_epoch + 1, 3000):
    loss = train(epoch)
    with open("./loss.txt", "a+") as f:
        f.write(str(epoch))
        f.write("\t")
        f.write(str(loss))
        f.write("\t")
        tim = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        f.write(str(tim))
        f.write("\n")

    # Save checkpoint
    if epoch % opt.num_cp == 0:
        model_save_path = join(model_dir, "model_epoch_{}.pth".format(epoch))
        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'losslogger': losslogger}
        torch.save(state, model_save_path)
        if min > loss:
            min = loss
            print(epoch)
            print("Checkpoint saved to {}".format(model_save_path))

    # Save loss snapshot
    if epoch % opt.num_snapshot == 0:
        plt.figure()
        plt.title('Loss')
        plt.plot(losslogger['epoch'], losslogger['loss'])
        plt.savefig(model_dir + ".jpg")
        plt.close()
