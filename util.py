import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset
import random
import cv2
from scipy.ndimage import gaussian_filter
import torch
import scipy.interpolate as spi
from torch.autograd import Variable
import tensorly as tl
from skimage.metrics import structural_similarity as compare_ssim
from torch import nn
from scipy.ndimage import rotate


def dot(m1, m2):
    r, c, b = m1.shape
    p = r * c
    temp_m1 = np.reshape(m1, [p, b], order='F')
    temp_m2 = np.reshape(m2, [p, b], order='F')
    out = np.zeros([p])
    for i in range(p):
        out[i] = np.inner(temp_m1[i, :], temp_m2[i, :])
    out = np.reshape(out, [r, c], order='F')
    return out


def CC(reference, target):
    bands = reference.shape[2]
    out = np.zeros([bands])
    for i in range(bands):
        ref_temp = reference[:, :, i].flatten(order='F')
        target_temp = target[:, :, i].flatten(order='F')
        cc = np.corrcoef(ref_temp, target_temp)
        out[i] = cc[0, 1]
    return np.mean(out)


def SAM(reference, target):
    rows, cols, bands = reference.shape
    pixels = rows * cols
    eps = 1 / (2 ** 52)  # 浮点精度
    prod_scal = dot(reference, target)  # 取各通道相同位置组成的向量进行内积运算
    norm_ref = dot(reference, reference)
    norm_tar = dot(target, target)
    prod_norm = np.sqrt(norm_ref * norm_tar)  # 二范数乘积矩阵
    prod_map = prod_norm
    prod_map[prod_map == 0] = eps  # 除法避免除数为0
    map = np.arccos(prod_scal / prod_map)  # 求得映射矩阵
    prod_scal = np.reshape(prod_scal, [pixels, 1])
    prod_norm = np.reshape(prod_norm, [pixels, 1])
    z = np.argwhere(prod_norm == 0)[:0]  # 求得prod_norm中为0位置的行号向量
    # 去除这些行，方便后续进行点除运算
    prod_scal = np.delete(prod_scal, z, axis=0)
    prod_norm = np.delete(prod_norm, z, axis=0)
    # 求取平均光谱角度
    angolo = np.sum(np.arccos(np.clip(prod_scal / prod_norm, -1, 1))) / prod_scal.shape[0]
    # 转换为度数
    angle_sam = np.real(angolo) * 180 / np.pi
    return angle_sam, map


def SSIM(reference, target):
    rows, cols, bands = reference.shape
    mssim = 0
    for i in range(bands):
        mssim += SSIM_BAND(reference[:, :, i], target[:, :, i])
    mssim /= bands
    return mssim


def SSIM_BAND(reference, target):
    return compare_ssim(reference, target, data_range=1.0)


def PSNR(reference, target):
    max_pixel = 1.0
    return 10.0 * np.log10((max_pixel ** 2) / np.mean(np.square(reference - target)))


# def PSNR(H_fuse, H_ref):
#     """
#     计算多光谱图像的平均PSNR，输入为numpy数组，shape为(H, W, C)
#     """
#     # 转换为 (C, H*W)
#     N_spectral = H_fuse.shape[2]
#     H_fuse_reshaped = H_fuse.transpose(2, 0, 1).reshape(N_spectral, -1)
#     H_ref_reshaped = H_ref.transpose(2, 0, 1).reshape(N_spectral, -1)

#     # 每个波段的RMSE
#     rmse = np.sqrt(np.sum((H_ref_reshaped - H_fuse_reshaped) ** 2, axis=1) / H_fuse_reshaped.shape[1])

#     # 每个波段的最大值
#     max_H_ref = np.max(H_ref_reshaped, axis=1)

#     # 每个波段的PSNR
#     psnr_band = 10 * np.log10((max_H_ref / rmse) ** 2 + 1e-8)  # 防止除零

#     # 平均PSNR
#     psnr_mean = np.nanmean(psnr_band)
#     return psnr_mean



def RMSE(reference, target):
    rows, cols, bands = reference.shape
    pixels = rows * cols * bands
    out = np.sqrt(np.sum((reference - target) ** 2) / pixels)
    return out


def ERGAS(references, target, ratio):
    rows, cols, bands = references.shape
    d = 1 / ratio
    pixels = rows * cols
    ref_temp = np.reshape(references, [pixels, bands], order='F')
    tar_temp = np.reshape(target, [pixels, bands], order='F')
    err = ref_temp - tar_temp
    rmse2 = np.sum(err ** 2, axis=0) / pixels
    uk = np.mean(tar_temp, axis=0)
    relative_rmse2 = rmse2 / uk ** 2
    total_relative_rmse = np.sum(relative_rmse2)
    out = 100 * d * np.sqrt(1 / bands * total_relative_rmse)
    return out


class Loss_SAM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, label, output):
        b, c, h, w = label.shape
        x_norm = torch.sqrt(torch.sum(torch.square(label), dim=1))
        y_norm = torch.sqrt(torch.sum(torch.square(output), dim=1))
        xy_norm = torch.multiply(x_norm, y_norm)
        xy = torch.sum(torch.multiply(label, output), dim=1)
        dist = torch.mean(torch.arccos(torch.minimum(torch.divide(xy, xy_norm + 1e-8), torch.tensor(1.0 - 1.0e-9))),
                          dim=[1, 2])
        dist = torch.multiply(torch.tensor(180.0 / np.pi), dist)
        dist = torch.mean(dist)
        return dist


def gauss_kernel(row_size, col_size, sigma):
    kernel = cv2.getGaussianKernel(row_size, sigma)
    kernel = kernel * cv2.getGaussianKernel(col_size, sigma).T
    return kernel


def anisotropic_gaussian_kernel(size, sigmaX, sigmaY, angle_degrees):
    center = size // 2
    kernel = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            kernel[x, y] = (1 / (2 * np.pi * sigmaX * sigmaY)) * np.exp(
                -((x - center) ** 2 / (2 * sigmaX ** 2) + (y - center) ** 2 / (2 * sigmaY ** 2)))
    kernel /= np.sum(kernel)

    # 旋转核
    rotated_kernel = rotate(kernel, angle_degrees, reshape=False, mode='constant', cval=0.0)
    return rotated_kernel


def intersect(list1, list2):
    list1 = list(list1)
    elem = list(set(list1).intersection(set(list2)))
    elem.sort()
    res = np.zeros(len(elem))
    for i in range(0, len(elem)):
        res[i] = list1.index(elem[i])
    res = res.astype("int32")
    return res


def create_spec_resp(data_num, genPath):
    if data_num == 0:  # CAVE  31 X 3
        band = 31
        file = os.path.join(genPath, 'srf/D700.mat')  # 377-948
        mat = sio.loadmat(file)
        spec_rng = np.arange(400, 700 + 1, 10)
        spec_resp = mat['spec_resp']
        #print("spec_rng:", spec_rng)
        R = spec_resp[spec_rng - 377, 1:4].T
    if data_num == 1:  # harvard  31 X 3
        file = os.path.join(genPath, 'srf/D700.mat')  # 377-948
        mat = sio.loadmat(file)
        spec_rng = np.arange(420, 720 + 1, 10)
        spec_resp = mat['spec_resp']
        R = spec_resp[spec_rng - 377, 1:4].T
    if data_num == 2:
        band = 102  # paviaC
        file = os.path.join(genPath, 'srf/ikonos.mat')  # 350 : 5 : 1035
        mat = sio.loadmat(file)
        spec_rng = np.arange(430, 861)
        spec_resp = mat['spec_resp']
        ms_bands = range(1, 5)
        valid_ik_bands = intersect(spec_resp[:, 0], spec_rng)
        no_wa = len(valid_ik_bands)
        # Spline interpolation
        xx = np.linspace(1, no_wa, band)
        x = range(1, no_wa + 1)
        R = np.zeros([5, band])
        for i in range(0, 5):
            ipo3 = spi.splrep(x, spec_resp[valid_ik_bands, i + 1], k=3)
            R[i, :] = spi.splev(xx, ipo3)
        R = R[ms_bands, :]

    if data_num == 3:
        # PaviaU 103 X 4
        band = 103 
        file = os.path.join(genPath, 'srf/ikonos_SRF.mat')
        mat = sio.loadmat(file)
        spec_resp = mat['R']
        R  = spec_resp

    if data_num == 4:
        # Chikusei  128 X 4
        band = 128
        file = os.path.join(genPath, 'srf/ikonos.mat')
        mat = sio.loadmat(file)
        spec_rng = np.arange(375, 1015, 5)
        spec_resp = mat['spec_resp']
        R = spec_resp[(spec_rng - 350) // 5, 2:6].T
    c = 1 / np.sum(R, axis=1)
    R = np.multiply(R, c.reshape([c.size, 1]))
    return R


def create_hrms_lrhs(hs, B, R, ratio, hs_snr, ms_snr, noise=True):
    hrms = tl.tenalg.mode_dot(hs, R, mode=2)
    # add noise for ms
    ms_sig = (np.sum(np.power(hrms.flatten(), 2)) / (10 ** (ms_snr / 10)) / hrms.size) ** 0.5
    np.random.seed(1)
    if noise is True:
        print('Add Noise')
        hrms = np.add(hrms, ms_sig * np.random.randn(hrms.shape[0], hrms.shape[1], hrms.shape[2]))
    # blur
    lrhs = cv2.filter2D(hs, -1, B, borderType=cv2.BORDER_REFLECT)
    # add noise for hs
    hs_sig = (np.sum(np.power(lrhs.flatten(), 2)) / (10 ** (hs_snr / 10)) / lrhs.size) ** 0.5
    np.random.seed(0)
    if noise is True:
        lrhs = np.add(lrhs, hs_sig * np.random.randn(lrhs.shape[0], lrhs.shape[1], lrhs.shape[2]))
    # down sampling
    lrhs = lrhs[0::ratio, 0::ratio, :]

    return hrms, lrhs


class degDataPreprocessing(object):
    def __init__(self, dataNum, genPath, factor):
        self.dataNum = dataNum
        self.genPath = genPath
        self.factor = factor
        self.R = torch.tensor(create_spec_resp(self.dataNum, self.genPath))

    def __call__(self, hr, rand=True):
        hr = hr.numpy()
        if rand:
            # hs_snr = 30
            # ms_snr = 40
            # sig_level = 9
            snr_level = random.randint(0, 2)
            if snr_level == 0:
                hs_snr, ms_snr = 25, 35
            elif snr_level == 1:
                hs_snr, ms_snr = 35, 45
            else:
                hs_snr, ms_snr = 40, 50
            sig_level = random.randint(1, 9)
            kerType = random.randint(5, 10)
        else:
            hs_snr = 30
            ms_snr = 40
            sig_level = 9
            kerType = 4
        sigma = (1 / (2 * 2.7725887 / sig_level ** 2)) ** 0.5
        kerSize = kerType * 2 - 1
        B = gauss_kernel(kerSize, kerSize, sigma=sigma)

        hr = hr.transpose(0, 1, 3, 4, 2)
        batch_size, N, H, W, C = hr.shape
        msi = np.zeros((batch_size, N, H, W, 3))
        hsi = np.zeros((batch_size, N, H // self.factor, W // self.factor, C))
        for i in range(batch_size):
            for j in range(N):
                img = hr[i, j, :, :, :]
                hrms, lrhs = create_hrms_lrhs(img, B, self.R, self.factor, hs_snr, ms_snr, noise=True)
                msi[i, j, :, :, :] = hrms
                hsi[i, j, :, :, :] = lrhs
        msi = msi.astype(np.float32)
        msi = torch.from_numpy(np.ascontiguousarray(msi.transpose(0, 1, 4, 2, 3)))
        hsi = hsi.astype(np.float32)
        hsi = torch.from_numpy(np.ascontiguousarray(hsi.transpose(0, 1, 4, 2, 3)))
        return msi, hsi







