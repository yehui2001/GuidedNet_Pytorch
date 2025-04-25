# -*- coding: utf-8 -*-
'''
@Author: Yehui
@Date: 2025-04-22 16:01:18
@LastEditTime: 2025-04-25 14:09:10
@FilePath: /GuidedNet/main.py
@Copyright (c) 2025 by , All Rights Reserved.
'''
import torch
import numpy as np
import os
from dataloader import TrainDataLoader,TestDataLoader
from torch.utils.data import DataLoader
import scipy.io as sio
import torch.optim as optim
from tqdm import tqdm  # 用于显示进度条
from net import HyNetSingleLevel,HyTestNet
from net_ import FRB
from util import PSNR,SAM,SSIM,ERGAS
import logging
from dataloader_single import get_single_image_datasets


global scale
global dataset
dataset = 'PaviaU'
scale = 8

# 日志配置
def log_generate(is_log:bool):
    log_dir = "./GuidedNet/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{dataset}.log")
    handlers = [logging.StreamHandler()]
    if is_log:
        handlers.append(logging.FileHandler(log_path, mode='a', encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=handlers
    )


# 训练函数
def train_epoch(model, loader, optimizer, device, alpha):
    model.train()  # 设置模型为训练模式
    total_loss = 0.0
    total_psnr = 0.0
    total_sam = 0.0
    total_ssim = 0.0
    total_ergas = 0.0

    for batch_gt, batch_ms,batch_rgb_hp in tqdm(loader, desc="Training"):

        # 将数据移动到指定设备（如 GPU）
        batch_gt = batch_gt.to(device)
        batch_rgb_hp = batch_rgb_hp.to(device)
        batch_ms = batch_ms.to(device)

        # 前向传播
        outputs = model(batch_ms, batch_rgb_hp)  # 假设模型返回多个尺度的输出
        net_image3, net_image2, net_image1 = outputs

        # 下采样目标图像以匹配不同尺度的输出
        target_down1 = torch.nn.functional.interpolate(
            batch_gt, scale_factor=1 / 2, mode='bicubic', align_corners=False
        )
        target_down2 = torch.nn.functional.interpolate(
            batch_gt, scale_factor=1 / 4, mode='bicubic', align_corners=False
        )

        # 计算多尺度损失
        loss1 = criterion(net_image3, batch_gt)
        loss2 = criterion(net_image2, target_down1)
        loss3 = criterion(net_image1, target_down2)
        loss = alpha[0] * loss1 + alpha[1] * loss2 + alpha[2] * loss3

        # 将输出的两个HR-HSI转为HWC的numpy，便于指标计算
        net_image3_numpy=net_image3.data.cpu().detach().numpy()[0].transpose(1,2,0)
        batch_gt_numpy=batch_gt.data.cpu().detach().numpy()[0].transpose(1,2,0)

        psnr = PSNR(batch_gt_numpy,net_image3_numpy)
        sam  = SAM(batch_gt_numpy,net_image3_numpy)
        ssim = SSIM(batch_gt_numpy,net_image3_numpy)
        ergas = ERGAS(batch_gt_numpy,net_image3_numpy,scale)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item()
        total_psnr += psnr
        total_sam  += sam[0]
        total_ssim += ssim
        total_ergas += ergas

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    avg_sam = total_sam / len(loader)
    avg_ssim = total_ssim / len(loader)
    avg_ergas = total_ergas/ len(loader)

    return avg_loss,avg_psnr,avg_sam,avg_ssim,avg_ergas


# 测试函数
def test_epoch(model, loader, device, alpha):
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    total_psnr = 0.0
    total_sam = 0.0   
    total_ssim = 0.0
    total_ergas = 0.0
    total_samples = 0.0

    with torch.no_grad():  # 不计算梯度
        for batch_gt, batch_ms,batch_rgb_hp in tqdm(loader, desc="Testing"):

            # 将数据移动到指定设备
            batch_gt = batch_gt.to(device)
            batch_rgb_hp = batch_rgb_hp.to(device)
            batch_ms = batch_ms.to(device)

            # 前向传播
            outputs = model(batch_ms, batch_rgb_hp)
            net_image3, net_image2, net_image1 = outputs

            # 下采样目标图像
            target_down1 = torch.nn.functional.interpolate(batch_gt, scale_factor=1 / 2, mode='bicubic', align_corners=False)
            target_down2 = torch.nn.functional.interpolate(batch_gt, scale_factor=1 / 4, mode='bicubic', align_corners=False)

            # 计算多尺度损失
            loss1 = criterion(net_image3, batch_gt)
            loss2 = criterion(net_image2, target_down1)
            loss3 = criterion(net_image1, target_down2)
            loss = alpha[0] * loss1 + alpha[1] * loss2 + alpha[2] * loss3

            # 将输出的两个HR-HSI转为HWC的numpy，便于指标计算
            net_image3_numpy=net_image3.data.cpu().detach().numpy()[0].transpose(1,2,0)
            batch_gt_numpy=batch_gt.data.cpu().detach().numpy()[0].transpose(1,2,0)

            # 计算相关指标
            psnr = PSNR(batch_gt_numpy,net_image3_numpy)
            sam  = SAM(batch_gt_numpy,net_image3_numpy)
            ssim = SSIM(batch_gt_numpy,net_image3_numpy)
            ergas = ERGAS(batch_gt_numpy,net_image3_numpy,scale)


            # # 累计损失
            # total_loss += loss.item()
            # total_psnr += psnr
            # total_sam  += sam[0]
            # total_ssim += ssim
            # total_ergas += ergas

            # 累加时乘以当前batch的样本数
            total_loss += loss.item() * batch_size
            total_psnr += psnr * batch_size
            total_sam  += sam[0] * batch_size
            total_ssim += ssim * batch_size
            total_ergas += ergas * batch_size
            total_samples += batch_size

        # # 取平均
        # avg_loss = total_loss / len(loader)
        # avg_psnr = total_psnr / len(loader)
        # avg_sam = total_sam / len(loader)
        # avg_ssim = total_ssim / len(loader)
        # avg_ergas = total_ergas/ len(loader)
        avg_loss = total_loss / total_samples
        avg_psnr = total_psnr / total_samples
        avg_sam = total_sam / total_samples
        avg_ssim = total_ssim / total_samples
        avg_ergas = total_ergas / total_samples
    return avg_loss,avg_psnr,avg_sam,avg_ssim,avg_ergas

if __name__ == '__main__':
    train_root = '/yehui/GuidedNet/dataset/CAVE/train'
    test_root = '/yehui/GuidedNet/dataset/CAVE/test'
    gen_path = '/yehui/GuidedNet/dataset/' # 响应函数路径
    root = '/yehui/GuidedNet/dataset/PaviaU'

    # 相关参数
    batch_size = 5 # CAVE:32 PAVIAU: 5
    in_patch_size = 10
    channel = 103
    alpha = [1, 2, 4]
    res_number = 10
    lr_rate = 0.0005

    train_dataset,test_dataset = get_single_image_datasets(root,gen_path)
    # 加载训练集
    #train_dataset = TrainDataLoader(root=train_root, genPath=gen_path)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    print(len(train_dataset))
    print(len(train_loader))

    # 加载测试集
    #test_dataset = TestDataLoader(root=test_root, genPath=gen_path)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)
    print(len(test_dataset))
    print(len(test_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = FRB(msi_c=4,hsi_c=103).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    save_dir = "./GuidedNet/saved_models"  # 保存模型的目录
    os.makedirs(save_dir, exist_ok=True)  # 创建保存模型的目录

    log_generate(1) # 选择记录log文件

    # 定义损失函数 (MSE Loss)
    # criterion1 = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()

    nums_epoch = 300
    # 主训练循环
    for epoch in range(nums_epoch):
        logging.info(f"Epoch [{epoch+1}/{nums_epoch}]    Learning rate:{lr_rate}")

        # 训练阶段
        train_loss,train_psnr,train_sam,train_ssim,train_ergas = train_epoch(model, train_loader, optimizer, device, alpha)
        logging.info(f"Train Loss: {train_loss:.6f} PSNR:{train_psnr:.6f} SAM:{train_sam:.6f} SSIM:{train_ssim:.6f} ERGAS:{train_ergas:.6f}")

        # 测试阶段
        test_loss,test_psnr,test_sam,test_ssim,test_ergas = test_epoch(model, test_loader, device, alpha)
        logging.info(f"Test Loss: {test_loss:.6f} PSNR:{test_psnr:.6f} SAM:{test_sam:.6f} SSIM:{test_ssim:.6f} ERGAS:{test_ergas:.6f}")

        # 保存模型 
        if (epoch + 1) % 50 == 0:  # 每 50 个 epoch 保存一次模型
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"Model saved to {save_path}")

        # 动态调整学习率
        if (epoch + 1) % 40 == 0 and lr_rate > 1e-6:
            lr_rate /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_rate
            logging.info(f"Learning rate updated to {lr_rate}")