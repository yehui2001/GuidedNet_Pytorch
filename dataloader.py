import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset
import torch
import glob
from util import create_hrms_lrhs, create_spec_resp, gauss_kernel


class TrainDataLoader(Dataset):
    #def __init__(self, root, genPath, factor=8, patch_size=16, kerSize=15, sigma=2.8): 此处的patch_size是lr_hsi的patch
    def __init__(self, root, genPath, factor=5, patch_size=10, kerSize=10, sigma=2): 
        super(TrainDataLoader, self).__init__()
        self.factor = factor
        self.hr_size = patch_size * factor
        self.lr_size = patch_size
        self.hs_snr = 30
        self.ms_snr = 40

        self.image_names = sorted(glob.glob(os.path.join(root, '*.mat')))
        self.sigma = sigma
        self.kerSize = kerSize
        self.B = gauss_kernel(self.kerSize, self.kerSize, self.sigma)
        self.R = create_spec_resp(0, genPath)
        self.hrhs_list = []
        self.lrhs_list = []
        self.hrms_list = []
        self.stride = 48 # 32
        for ind in range(len(self.image_names)): # 遍历每一张照片
            hrhs = sio.loadmat(self.image_names[ind])
            hrhs = hrhs["HS"].astype(np.float32)
            #hrhs = hrhs["orig"].astype(np.float32)
            hrms, lrhs = create_hrms_lrhs(hrhs, self.B, self.R, self.factor, self.hs_snr, self.ms_snr, noise=False)
            H, W, C = hrhs.shape
            #print(H,W)

            # 计算patch的数量
            n_rows = (H - self.hr_size) // self.stride + 1
            n_cols = (W - self.hr_size) // self.stride + 1
            #print(n_rows,n_cols)

            # 遍历图像并提取patch
            for i in range(n_rows):
                for j in range(n_cols):
                    # 计算当前patch的左上角坐标
                    hr_row = i * self.stride
                    hr_col = j * self.stride
                    lr_row = i * self.stride // self.factor
                    lr_col = j * self.stride // self.factor
                    # 提取当前patch
                    hrhs_patch = hrhs[hr_row: hr_row + self.hr_size, hr_col: hr_col + self.hr_size, :]
                    hrms_patch = hrms[hr_row: hr_row + self.hr_size, hr_col: hr_col + self.hr_size, :]
                    lrhs_patch = lrhs[lr_row: lr_row + self.lr_size, lr_col: lr_col + self.lr_size, :]

                    self.hrhs_list.append(hrhs_patch)
                    self.hrms_list.append(hrms_patch)
                    self.lrhs_list.append(lrhs_patch)



    def __len__(self):
        return len(self.hrhs_list)

    def __getitem__(self, index):
        hrhs, hrms, lrhs = self.hrhs_list[index], self.hrms_list[index], self.lrhs_list[index]
        hrhs = hrhs.astype(np.float32)
        hrms = hrms.astype(np.float32)
        lrhs = lrhs.astype(np.float32)
        hrhs = torch.from_numpy(np.ascontiguousarray(hrhs.transpose(2, 0, 1)))
        hrms = torch.from_numpy(np.ascontiguousarray(hrms.transpose(2, 0, 1)))
        lrhs = torch.from_numpy(np.ascontiguousarray(lrhs.transpose(2, 0, 1)))

        return hrhs, lrhs, hrms


class TestDataLoader(Dataset):
    #def __init__(self, root, genPath, factor=8, patch_size=8, kerSize=15, sigma=2.8):
    def __init__(self, root, genPath, factor=8, patch_size=10, kerSize=10, sigma=2):
        super(TestDataLoader, self).__init__()
        self.factor = factor
        self.hs_snr = 30
        self.ms_snr = 40
        self.hr_size = patch_size * factor
        self.lr_size = patch_size
        self.sigma = sigma
        self.kerSize = kerSize
        self.B = gauss_kernel(self.kerSize, self.kerSize, self.sigma)
        self.R = create_spec_resp(0, genPath)
        self.image_names = sorted(glob.glob(os.path.join(root, '*.mat')))
        self.hrhs_list = []
        self.lrhs_list = []
        self.hrms_list = []
        self.stride = 48#48
        for ind in range(len(self.image_names)):
            hrhs = sio.loadmat(self.image_names[ind])
            #hrhs = hrhs["HS"].astype(np.float32)
            hrhs = hrhs["HS"].astype(np.float32)
            hrms, lrhs = create_hrms_lrhs(hrhs, self.B, self.R, self.factor, self.hs_snr, self.ms_snr, noise=False)
            H, W, C = hrhs.shape

            # 计算patch的数量
            n_rows = (H - self.hr_size) // self.stride + 1
            n_cols = (W - self.hr_size) // self.stride + 1

            # 遍历图像并提取patch
            for i in range(n_rows):
                for j in range(n_cols):
                    # 计算当前patch的左上角坐标
                    hr_row = i * self.stride
                    hr_col = j * self.stride
                    lr_row = i * self.stride // self.factor
                    lr_col = j * self.stride // self.factor
                    # 提取当前patch
                    hrhs_patch = hrhs[hr_row: hr_row + self.hr_size, hr_col: hr_col + self.hr_size, :]
                    hrms_patch = hrms[hr_row: hr_row + self.hr_size, hr_col: hr_col + self.hr_size, :]
                    lrhs_patch = lrhs[lr_row: lr_row + self.lr_size, lr_col: lr_col + self.lr_size, :]

                    self.hrhs_list.append(hrhs_patch)
                    self.hrms_list.append(hrms_patch)
                    self.lrhs_list.append(lrhs_patch)

    def __len__(self):
        return len(self.hrhs_list)

    def __getitem__(self, index):
        hrhs, hrms, lrhs = self.hrhs_list[index], self.hrms_list[index], self.lrhs_list[index]
        hrhs = hrhs.astype(np.float32)
        hrms = hrms.astype(np.float32)
        lrhs = lrhs.astype(np.float32)

        hrhs = torch.from_numpy(np.ascontiguousarray(hrhs.transpose(2, 0, 1)))
        hrms = torch.from_numpy(np.ascontiguousarray(hrms.transpose(2, 0, 1)))
        lrhs = torch.from_numpy(np.ascontiguousarray(lrhs.transpose(2, 0, 1)))
        return hrhs, lrhs, hrms # GT , MS , RGB_HP

