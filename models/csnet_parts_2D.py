# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:48:44 2022

@author: Qi You
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.interp = nn.functional.interpolate()
    def forward(self, x):
        
        x = F.interpolate(x,(256,128,1600),mode='trilinear')
        x = F.normalize(x,2,)
        return torch.squeeze(self.input_conv(x),1)


class InputConv(nn.Module):

    def __init__(self, window_len = 32, sliding_len = 4):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(window_len,3,3), stride=(sliding_len, 1, 1), padding=(window_len//2-2, 1, 1), bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        return torch.squeeze(self.input_conv(x),1)


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, outUpScale=(2,2), bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
        self.outUp = nn.Upsample(scale_factor=outUpScale, mode='bilinear', align_corners=True)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        self.matchUp = nn.Upsample(size=(x1.size()[2],x1.size()[3]), mode='bilinear', align_corners=True)
        x2 = self.matchUp(x2)
        

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        return self.outUp(x)

class Up2(nn.Module):


    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()


        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        return x

class OutUp(nn.Module):

    def __init__(self, in_channels, out_channels, outUpScale=(1,2), bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
          
        self.outUp = nn.Upsample(scale_factor=outUpScale, mode='bilinear', align_corners=True)    
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        self.matchUp = nn.Upsample(size=(x1.size()[2],x1.size()[3]), mode='bilinear', align_corners=True)
        x2 = self.matchUp(x2)
        

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        return self.outUp(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.outConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.outConv(x)
