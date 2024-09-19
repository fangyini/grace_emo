import os
import glob

import numpy as np

dataset_path = 'grace_emo/dataset/processed_gau_600/'
index_path = os.path.join(dataset_path, 'index_files')
if not os.path.exists(index_path):
    os.makedirs(index_path)

dataset_size = len(glob.glob(os.path.join(dataset_path, 'data/*.npz')))
index_list = [i for i in range(dataset_size)]
np.random.shuffle(index_list)
train_list = index_list[:int(dataset_size * 0.8)]
val_list = index_list[int(dataset_size * 0.8):int(dataset_size * 0.9)]
test_list = index_list[int(dataset_size * 0.9):]

import pickle

with open(os.path.join(index_path, 'train.txt'), 'wb') as fp:
    pickle.dump(train_list, fp)
with open(os.path.join(index_path, 'val.txt'), 'wb') as fp:
    pickle.dump(val_list, fp)
with open(os.path.join(index_path, 'test.txt'), 'wb') as fp:
    pickle.dump(test_list, fp)
