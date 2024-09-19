import glob

import numpy as np
import os
import random
import torch
from torch.utils.data import DataLoader
from emotion.motor_prediction.utils import get_device

device = get_device()

class GraceFaceDataset():
    def __init__(self, split, image_path='/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/dataset/processed_gau/'):
        filenames = sorted(glob.glob(os.path.join(image_path, 'data_*.npz')))
        total_len = len(filenames)
        random.shuffle(filenames)
        self.split = split
        if split == 'train':
            self.data = filenames[:int(total_len*0.8)]
        elif split == 'val':
            self.data = filenames[int(total_len*0.8):int(total_len*0.9)]
        elif split == 'test':
            self.data = filenames[int(total_len*0.9):]
        print('len=', len(self.data))
        print('Images are currently not used!')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]
        data = np.load(filename)
        image = data['image'].astype(np.float32)
        landmarks = data['ldmk'].flatten().astype(np.float32)
        label = data['label'].flatten().astype(np.float32)
        if self.split == 'test':
            return image, landmarks, label, filename
        else:
            return image, landmarks, label

if __name__ == "__main__":
    dataset = GraceFaceDataset()
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for data in train_dataloader:
       pass