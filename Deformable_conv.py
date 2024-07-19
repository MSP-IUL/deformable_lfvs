# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 19:31:11 2024

@author: MSP
"""

import torch
import torchvision.ops
from torch import nn

class DeformableConv2d(nn.Module):
    """
    A class to implement deformable convolution, which allows for learning offsets
    and modulations to the regular convolution operation, enabling the network to handle
    geometric transformations more effectively.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        self.padding = padding
        
        # Convolution layer to predict the offsets for deformable convolution
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)
        # Initialize offset convolution weights and biases to zero
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        # Convolution layer to predict the modulator for deformable convolution
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)
        # Initialize modulator convolution weights and biases to zero
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        # Regular convolution layer
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)
    
    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.

        # Calculate offsets and clamp them to a maximum value
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        # Calculate modulator and scale it between 0 and 2
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        # Apply deformable convolution
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator)
        return x
