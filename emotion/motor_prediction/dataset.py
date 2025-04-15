import glob

import numpy as np
import os
import random
import torch
from torch.utils.data import DataLoader
from emotion.motor_prediction.utils import get_device
import pickle
device = get_device()

class LandmarkNormalize:
    """Normalize landmarks by dividing by 112"""
    def __call__(self, x):
        return x / 112.0

class GraceFaceDataset():
    def __init__(self, split, image_path='/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/dataset/processed_gau/', feature_type='face_embed'):
        filenames = sorted(glob.glob(os.path.join(image_path, 'data/data_*.npz')))
        total_len = len(filenames)
        random.shuffle(filenames)
        self.split = split
        self.feature_type = feature_type
        self.ldmk_transform = LandmarkNormalize()
        '''if split == 'train':
            self.data = filenames[:int(total_len*0.8)]
        elif split == 'val':
            self.data = filenames[int(total_len*0.8):int(total_len*0.9)]
        elif split == 'test':
            self.data = filenames[int(total_len*0.9):]'''

        index_list_file = os.path.join(image_path, 'index_files/' + str(split) + '.txt')
        with open(index_list_file, 'rb') as fp:
            index_list = pickle.load(fp)

        self.split = split
        self.data = []
        for h in index_list:
            self.data.append(filenames[h])

        print('len=', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]
        data = np.load(filename)
        if self.feature_type == 'face_embed':
            features = data['face_embed'].flatten().astype(np.float32)
        else:  # ldmk
            features = self.ldmk_transform(data['ldmk'].flatten().astype(np.float32))
        label = data['label'].flatten().astype(np.float32)
        if self.split == 'test':
            return features, label, filename
        else:
            return features, label

if __name__ == "__main__":
    dataset = GraceFaceDataset()
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for data in train_dataloader:
       pass