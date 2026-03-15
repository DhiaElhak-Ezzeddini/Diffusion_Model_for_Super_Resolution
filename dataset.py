import torch
import os
from tifffile import tifffile

class BioSRDDataset(torch.utils.data.Dataset):
    def __init__(self, lr_paths, hr_paths, transform=None):
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.transform = transform
        

        self.filename_list = [file for file in os.listdir(self.lr_paths) if file.endswith('.tif')]

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        lr_image_path = os.path.join(self.lr_paths,self.filename_list[idx])
        hr_image_path = os.path.join(self.hr_paths,self.filename_list[idx])
        lr_image = tifffile.imread(lr_image_path)
        hr_image = tifffile.imread(hr_image_path)

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        return lr_image, hr_image