# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:48:44 2022

@author: Qi You
"""


from .csnet_parts_2D import *
import torch.nn as nn


class CSNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256, (2,2), bilinear)
        self.up2 = Up(256, 128, (2,2), bilinear)
        self.up3 = OutUp(128, 64, (1,2), bilinear)
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
    