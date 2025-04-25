# -*- coding: utf-8 -*-
'''
@Author: Yehui
@Date: 2025-04-22 16:08:08
@LastEditTime: 2025-04-23 10:56:50
@FilePath: /GuidedNet/net_.py
@Copyright (c) 2025 by , All Rights Reserved.
'''
import os
from re import S
from turtle import forward, st
import torch
import torch.nn as nn


class DFM(nn.Module):
    def __init__(self, msi_c=3, hsi_c=31, feature_c=64, nums_res=10):
        super().__init__()
        self.nums_res = nums_res
        # 采用转置卷积进行上采样->output = (input-1)*stride - 2*padding + kernel_size + outputpadding
        self.upsample = nn.ConvTranspose2d(
            feature_c, feature_c, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # Output = 2 * Input
        self.concat_conv1 = nn.Conv2d(
            feature_c + msi_c, feature_c, kernel_size=3, stride=1, padding="same"
        )
        self.concat_conv2 = nn.Conv2d(
            feature_c + hsi_c, hsi_c, kernel_size=3, stride=1, padding="same"
        )
        self.pixelShuffle = nn.Sequential(
            nn.Conv2d(hsi_c, 4 * hsi_c, kernel_size=3, stride=1, padding="same"), 
            nn.PixelShuffle(2)
        )
        self.resnet_erb = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    feature_c, feature_c, kernel_size=3, stride=1, padding="same"
                ),
            ) for _ in range(self.nums_res)
        ])
    def forward(self,input_feature,input_lrhsi,input_msi):
        x1 = self.upsample(input_feature)
        x1 = torch.cat((x1, input_msi), dim=1)
        x1 = self.concat_conv1(x1)
        for block in self.resnet_erb: # 要循环累加，而不是直接遍历Sequential
            tmp = block(x1)
            x1  = x1 + tmp
        x2 = self.pixelShuffle(input_lrhsi)
        x3 = torch.cat((x2, x1), dim=1)
        x3 = self.concat_conv2(x3)
        return x1,x3
    
class FRB(nn.Module):
    def __init__(self, msi_c=3, hsi_c=31, feature_c = 64,nums_res = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(hsi_c, feature_c, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(feature_c, feature_c, kernel_size=3, stride=1, padding='same')
        # 下采样将尺寸减半 
        self.down_samples1 = nn.Conv2d(msi_c, msi_c, kernel_size=4, stride=2, padding=1)
        self.down_samples2 = nn.Conv2d(msi_c, msi_c, kernel_size=4, stride=2, padding=1)
        self.dfm1 = DFM(msi_c, hsi_c, feature_c, nums_res)
        self.dfm2 = DFM(msi_c, hsi_c, feature_c, nums_res)
        self.dfm3 = DFM(msi_c, hsi_c, feature_c, nums_res)

    def forward(self,lrhsi,hrmsi):
        input_feature = self.conv1(lrhsi)
        input_feature = self.conv2(input_feature)
        hrmsi_3 = hrmsi
        hrmsi_2 = self.down_samples1(hrmsi_3)
        hrmsi_1 = self.down_samples2(hrmsi_2)
        x_feature,x_up1 = self.dfm1(input_feature,lrhsi,hrmsi_1)
        x_feature,x_up2 = self.dfm2(x_feature,x_up1,hrmsi_2)
        x_feature,x_up3 = self.dfm3(x_feature,x_up2,hrmsi_3)
        return x_up3, x_up2, x_up1 

if __name__ == '__main__':
    model = FRB()
    X = torch.rand(1, 64, 10, 10)
    Y = torch.rand(1, 31,10, 10)
    Z = torch.rand(1,3,80,80)
    _,_,x_pred = model(Y,Z)
    print(x_pred.shape)



           
