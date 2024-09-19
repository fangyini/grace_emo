import torch
import glob
import os
import numpy as np
import pickle
def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device

def calculate_data_stat(path='/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/dataset/processed_gau/'):
    filenames = sorted(glob.glob(os.path.join(path, 'data/data_*.npz')))

    index_list_file = os.path.join(path, 'index_files/train.txt')
    with open(index_list_file, 'rb') as fp:
        index_list = pickle.load(fp)
    new_filenames = []
    for h in index_list:
        new_filenames.append(filenames[h])

    ldmk_list = []
    label_list = []
    for filename in new_filenames:
        data = np.load(filename)
        landmarks = data['ldmk'].flatten().astype(np.float32)
        label = data['label'].flatten().astype(np.float32)
        ldmk_list.append(landmarks)
        label_list.append(label)
    ldmk_list = np.stack(ldmk_list)
    label_list = np.stack(label_list)
    label_mean, label_std = np.mean(label_list), np.std(label_list)
    ldmk_mean, ldmk_std = np.mean(ldmk_list), np.std(ldmk_list)
    return label_mean, label_std, ldmk_mean, ldmk_std

