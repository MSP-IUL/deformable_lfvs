# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 19:32:51 2024

@author: MSP
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import einops
from Deformable_conv import DeformableConv2d

class Net(nn.Module):
    """
    Main network class that defines the model architecture.
    """
    def __init__(self, opt):
        super(Net, self).__init__()
        num = opt.num_source
        an = opt.angular_out
        an2 = opt.angular_out * opt.angular_out
        
        # Disparity estimation using deformable convolutions
        self.disp_estimator = nn.Sequential(
            DeformableConv2d(in_channels=opt.num_source, out_channels=16, kernel_size=11, stride=1, padding=5),
            nn.ReLU(inplace=True),
            DeformableConv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            DeformableConv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            DeformableConv2d(in_channels=64, out_channels=an2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            DeformableConv2d(in_channels=an2, out_channels=an2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DeformableConv2d(in_channels=an2, out_channels=an2, kernel_size=3)
        )

        # Lenslet feature extraction
        self.up_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(6, 1, 1), dilation=(2, 1, 1), groups=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(6, 1, 1), dilation=(3, 1, 1), groups=4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(5, 2, 2), dilation=(5, 1, 1), groups=8),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(1, 2, 2), groups=16),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=16),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=16)
        )
        self.pointwise = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        # Feature extraction network
        self.feature_extraction = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 7, 7), stride=(1, 1, 1), padding=(1, 6, 6), dilation=(1, 2, 2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 4, 4), dilation=(1, 2, 2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=32, out_channels=an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=an2, out_channels=an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=an2, out_channels=an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=an2, out_channels=an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )

        # Residual learning layers
        self.residual_learning_2 = make_ResBlock_2(layer_num=4, an2=an2, kernel_size=(3, 3, 3))

        # View synthesis layer
        self.view_synthesis = nn.Sequential(
            nn.Conv3d(in_channels=an2, out_channels=an2, kernel_size=(4, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        )

        # Feature fusion network
        self.lens_conv = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=6, kernel_size=3, stride=1, padding=1)
        )

        # Additional layers for angular refinement
        self.ansp = nn.Conv3d(in_channels=7, out_channels=7, kernel_size=1, stride=1)
        self.hang_conv = nn.Sequential(
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=an),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=an),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=an),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=an)
        )
        self.hang_pointwise = nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        # Additional residual learning layers
        self.Fuse = make_FuseBlock(layer_num=4, a=49, kernel_size=(3, 3))
        self.residual_learning = make_ResBlock_1(layer_num=4, an2=an2, kernel_size=(3, 3, 3))
        self.residual_learning_1 = make_ResBlock(layer_num=2, an2=6, kernel_size=(3, 3, 3))
        self.residual_learning_3 = make_ResBlock_3(layer_num=2, an2=6, kernel_size=(3, 3, 3))

    def forward(self, ind_source, img_source, LFI, opt):
        N, num_source, h, w = img_source.shape
        an = opt.angular_out
        an2 = opt.angular_out * opt.angular_out

        # Process input images and extract features
        D3_img = img_source.view(1, N, num_source, h, w)
        D3_img = self.feature_extraction(D3_img)
        D3_img = self.residual_learning_2(D3_img)
        D3_img = self.view_synthesis(D3_img)
        D3_img = D3_img.view(an2, N, h, w)

        # Estimate disparities
        disp_target = self.disp_estimator(img_source)

        # Warp input images based on estimated disparities
        ind_source = torch.squeeze(ind_source)
        warp_img_input = img_source.view(N * num_source, 1, h, w)
        warp_img_input = warp_img_input.repeat(an2, 1, 1, 1)
        grid = []
        for k_t in range(an2):
            for k_s in range(num_source):
                ind_s = ind_source[k_s].type_as(img_source)
                ind_t = torch.arange(an2)[k_t].type_as(img_source)
                ind_s_h = torch.floor(ind_s / an)
                ind_s_w = ind_s % an
                ind_t_h = torch.floor(ind_t / an)
                ind_t_w = ind_t % an
                disp = disp_target[:, k_t, :, :]
                XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(img_source)
                YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(img_source)
                grid_w_t = XX + disp * (ind_t_w - ind_s_w)
                grid_h_t = YY + disp * (ind_t_h - ind_s_h)
                grid_w_t_norm = 2.0 * grid_w_t / (w - 1) - 1.0
                grid_h_t_norm = 2.0 * grid_h_t / (h - 1) - 1.0
                grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm), dim=3)
                grid.append(grid_t)
        grid = torch.cat(grid, 0)
        warped_img_view = functional.grid_sample(warp_img_input, grid, align_corners=True).view(N, an2, num_source, h, w)

        # Process lenslet images
        LFI = LFI.reshape(-1, 1, 1, 2, 2)
        LFI = LFI.permute(1, 2, 0, 3, 4)
        LFI = self.up_conv(LFI)
        LFI = self.pointwise(LFI)
        LFI = torch.squeeze(LFI, 0)
        LFI = LFI.permute(1, 0, 2, 3)
        pixel_shuffle = nn.PixelShuffle(4)
        LFI = pixel_shuffle(LFI)
        lens_data = LFI.reshape(1, h, w, 8, 8)

        # Convert lenslet images to SAI format
        img = []
        allah = an
        sub = torch.zeros((1, h, w))
        for v in range(allah):
            for u in range(allah):
                sub[:, 0:h, 0:w] = lens_data[:, :, :, v, u]
                img.append(sub[:, :, :])
        img = torch.stack(img).cuda()
        img = img.permute(1, 0, 2, 3)
        img = self.Fuse(img)
        img = img.permute(1, 0, 2, 3)

        # Concatenate warped images and features
        warped_img_1 = warped_img_view.reshape(an2, 4, h, w)
        warped_0 = torch.cat((warped_img_1, D3_img, img), dim=1)
        warped_0 = self.lens_conv(warped_0)
        warped_0 = warped_0.view(1, an2, 6, h, w)
        warped_all = self.residual_learning(warped_0)
        warped_0 = warped_all.view(an2, 6, h, w)
        warped_h = warped_0.view(7, 7, 6, h, w)
        ansp = self.ansp(warped_h)
        hang_u = self.hang_conv(ansp)
        hang_v = self.hang_conv(ansp)
        hang_u = self.hang_pointwise(hang_u)
        hang_v = self.hang_pointwise(hang_v)
        hang_u = einops.rearrange(hang_u, "b anout an h w -> (b anout) an h w")
        hang_v = einops.rearrange(hang_v, "b anout an h w -> (b anout) an h w")
        hang_u = hang_u.view(1, an2, 6, h, w)
        hang_v = hang_v.view(1, an2, 6, h, w)
        hang_u = torch.transpose(hang_u, 1, 2)
        hang_v = torch.transpose(hang_v, 1, 2)
        hang_u = self.residual_learning_1(hang_u)
        hang_v = self.residual_learning_1(hang_v)
        hang_v = hang_v.squeeze(0)
        hang_u = hang_u.squeeze(0)
        hang_v = torch.transpose(hang_v, 0, 1)
        hang_u = torch.transpose(hang_u, 0, 1)
        hang_u = self.conv_1(hang_u)
        hang_v = self.conv_1(hang_v)
        hang_v = torch.transpose(hang_v, 0, 1)
        hang_u = torch.transpose(hang_u, 0, 1)

        # Sum features to get final output
        warped_fuse = warped_all[:, :, 0, :, :] + warped_all[:, :, 1, :, :] + warped_all[:, :, 2, :, :] + warped_all[:, :, 3, :, :] + warped_all[:, :, 4, :, :] + warped_all[:, :, 5, :, :]
        lf = warped_fuse + hang_v + hang_u
        
        return warped_img_view, lf

class ResBlock(nn.Module):
    """
    Basic residual block with 3D convolutions.
    """
    def __init__(self, an2, kernel_size, i):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self._initialize_weights(an2, kernel_size, i)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = x + out
        return out

    def _initialize_weights(self, an2, kernel_size, i):
        if i == 0:
            sigma = math.sqrt(2 / (1 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        else:
            sigma = math.sqrt(2 / (an2 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(std=sigma)
                m.bias.data.fill_(0)

def make_ResBlock(layer_num, an2, kernel_size):
    """
    Function to create a sequential block of residual layers.
    """
    layers = []
    for i in range(layer_num):
        layers.append(ResBlock(an2, kernel_size, i))
    return nn.Sequential(*layers)

class FuseBlock(nn.Module):
    """
    Basic residual block with 2D convolutions for feature fusion.
    """
    def __init__(self, a, kernel_size, i):
        super(FuseBlock, self).__init__()
        self.conv1 = nn.Conv2d(a, a, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(a, a, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1))
        self._initialize_weights(a, kernel_size, i)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = x + out
        return out

    def _initialize_weights(self, a, kernel_size, i):
        if i == 0:
            sigma = math.sqrt(2 / (1 * a * kernel_size[0] * kernel_size[1]))
        else:
            sigma = math.sqrt(2 / (a * a * kernel_size[0] * kernel_size[1]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=sigma)
                m.bias.data.fill_(0)

def make_FuseBlock(layer_num, a, kernel_size):
    """
    Function to create a sequential block of feature fusion layers.
    """
    layers = []
    for i in range(layer_num):
        layers.append(FuseBlock(a, kernel_size, i))
    return nn.Sequential(*layers)

class ResBlock_1(nn.Module):
    """
    Residual block with 3D convolutions, another variant.
    """
    def __init__(self, an2, kernel_size, i):
        super(ResBlock_1, self).__init__()
        self.conv1 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self._initialize_weights(an2, kernel_size, i)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(x)
        out = x + out
        return out

    def _initialize_weights(self, an2, kernel_size, i):
        if i == 0:
            sigma = math.sqrt(2 / (1 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        else:
            sigma = math.sqrt(2 / (an2 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(std=sigma)
                m.bias.data.fill_(0)

def make_ResBlock_1(layer_num, an2, kernel_size):
    """
    Function to create a sequential block of ResBlock_1 layers.
    """
    layers = []
    for i in range(layer_num):
        layers.append(ResBlock_1(an2, kernel_size, i))
    return nn.Sequential(*layers)

class ResBlock_2(nn.Module):
    """
    Residual block with 3D convolutions, another variant.
    """
    def __init__(self, an2, kernel_size, i):
        super(ResBlock_2, self).__init__()
        self.conv1 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self._initialize_weights(an2, kernel_size, i)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(x)
        out = x + out
        return out

    def _initialize_weights(self, an2, kernel_size, i):
        if i == 0:
            sigma = math.sqrt(2 / (1 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        else:
            sigma = math.sqrt(2 / (an2 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(std=sigma)
                m.bias.data.fill_(0)

def make_ResBlock_2(layer_num, an2, kernel_size):
    """
    Function to create a sequential block of ResBlock_2 layers.
    """
    layers = []
    for i in range(layer_num):
        layers.append(ResBlock_2(an2, kernel_size, i))
    return nn.Sequential(*layers)

class ResBlock_3(nn.Module):
    """
    Residual block with 3D convolutions, another variant.
    """
    def __init__(self, an2, kernel_size, i):
        super(ResBlock_3, self).__init__()
        self.conv1 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self._initialize_weights(an2, kernel_size, i)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(x)
        out = x + out
        return out

    def _initialize_weights(self, an2, kernel_size, i):
        if i == 0:
            sigma = math.sqrt(2 / (1 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        else:
            sigma = math.sqrt(2 / (an2 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(std=sigma)
                m.bias.data.fill_(0)

def make_ResBlock_3(layer_num, an2, kernel_size):
    """
    Function to create a sequential block of ResBlock_3 layers.
    """
    layers = []
    for i in range(layer_num):
        layers.append(ResBlock_3(an2, kernel_size, i))
    return nn.Sequential(*layers)
