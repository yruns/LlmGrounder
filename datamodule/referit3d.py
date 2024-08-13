"""
File: referit3d.py
Date: 2024/8/6
Author: yruns


"""
from torch.utils.data import Dataset, DataLoader


class ReferIt3D(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return idx
