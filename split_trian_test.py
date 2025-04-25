import os
import random
import shutil

all_data_path = '/yehui/datasets/CAVE'
train_path = '/yehui/GuidedNet/dataset/train'
test_path = '/yehui/GuidedNet/dataset/test'

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

mat_files = [f for f in os.listdir(all_data_path) if f.endswith('.mat')]
random.shuffle(mat_files) # 随机打乱文件列表顺序

split_idx = 21            #int(len(mat_files) * 0.8)
train_files = mat_files[:split_idx]
test_files = mat_files[split_idx:]

for f in train_files:
    shutil.copy(os.path.join(all_data_path, f), os.path.join(train_path, f))
for f in test_files:
    shutil.copy(os.path.join(all_data_path, f), os.path.join(test_path, f))

print(f"划分完成：训练集 {len(train_files)} 张，测试集 {len(test_files)} 张")