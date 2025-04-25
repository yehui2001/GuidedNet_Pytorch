import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset
import torch
import glob
from util import create_hrms_lrhs, create_spec_resp, gauss_kernel

def split_patches(hrhs_list, hrms_list, lrhs_list, split_ratio=0.8):
    total = len(hrhs_list)
    indices = np.arange(total)
    np.random.shuffle(indices)
    split = int(total * split_ratio)
    train_idx, test_idx = indices[:split], indices[split:]
    train = ([hrhs_list[i] for i in train_idx],
             [hrms_list[i] for i in train_idx],
             [lrhs_list[i] for i in train_idx])
    test = ([hrhs_list[i] for i in test_idx],
            [hrms_list[i] for i in test_idx],
            [lrhs_list[i] for i in test_idx])
    return train, test

class SingleImagePatchDataset(Dataset):
    def __init__(self, hrhs_list, hrms_list, lrhs_list):
        self.hrhs_list = hrhs_list
        self.hrms_list = hrms_list
        self.lrhs_list = lrhs_list

    def __len__(self):
        return len(self.hrhs_list)

    def __getitem__(self, index):
        hrhs, hrms, lrhs = self.hrhs_list[index], self.hrms_list[index], self.lrhs_list[index]
        hrhs = torch.from_numpy(np.ascontiguousarray(hrhs.astype(np.float32).transpose(2, 0, 1)))
        hrms = torch.from_numpy(np.ascontiguousarray(hrms.astype(np.float32).transpose(2, 0, 1)))
        lrhs = torch.from_numpy(np.ascontiguousarray(lrhs.astype(np.float32).transpose(2, 0, 1)))
        return hrhs, lrhs, hrms

def get_single_image_datasets(root, genPath, factor=8, patch_size=8, kerSize=3, sigma=2, split_ratio=0.8):
    image_names = sorted(glob.glob(os.path.join(root, '*.mat')))
    assert len(image_names) > 0, "No .mat files found!"
    hrhs = sio.loadmat(image_names[0])["HS"].astype(np.float32)
    B = gauss_kernel(kerSize, kerSize, sigma)
    R = create_spec_resp(3, genPath)
    hrms, lrhs = create_hrms_lrhs(hrhs, B, R, factor, 30, 40, noise=False)
    H, W, C = hrhs.shape
    hr_size = patch_size * factor
    lr_size = patch_size
    stride = 16

    hrhs_list, hrms_list, lrhs_list = [], [], []
    n_rows = (H - hr_size) // stride + 1
    n_cols = (W - hr_size) // stride + 1
    for i in range(n_rows):
        for j in range(n_cols):
            hr_row = i * stride
            hr_col = j * stride
            lr_row = i * stride // factor
            lr_col = j * stride // factor
            hrhs_patch = hrhs[hr_row: hr_row + hr_size, hr_col: hr_col + hr_size, :]
            hrms_patch = hrms[hr_row: hr_row + hr_size, hr_col: hr_col + hr_size, :]
            lrhs_patch = lrhs[lr_row: lr_row + lr_size, lr_col: lr_col + lr_size, :]
            hrhs_list.append(hrhs_patch)
            hrms_list.append(hrms_patch)
            lrhs_list.append(lrhs_patch)

    (train_hrhs, train_hrms, train_lrhs), (test_hrhs, test_hrms, test_lrhs) = split_patches(
        hrhs_list, hrms_list, lrhs_list, split_ratio=split_ratio
    )

    train_dataset = SingleImagePatchDataset(train_hrhs, train_hrms, train_lrhs)
    test_dataset = SingleImagePatchDataset(test_hrhs, test_hrms, test_lrhs)
    return train_dataset, test_dataset

if __name__ == '__main__':
    gen_path = '/yehui/GuidedNet/dataset/' # 响应函数路径
    root = '/yehui/GuidedNet/dataset/PaviaU'
    train_dataset, test_dataset = get_single_image_datasets(root, gen_path)
    print(len(train_dataset))
    print(len(test_dataset))