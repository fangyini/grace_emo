import os
import glob
from tqdm import tqdm
import csv
import numpy as np
import argparse

def read_csv(csv_path):
    res = {}
    replaced_num = 0
    with open(csv_path, "r", encoding="utf-8", newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter="\t")
        for row in csv_reader:
            filename, label = row
            label = label[1:-1].split(', ')
            label = np.array([float(i) for i in label])
            assert label.shape[0] == 26
            if filename in res:
                replaced_num += 1
                #print('repeated rows in label csv, replacing ', filename)
            res[filename] = label
    print('Total replaced num: ', replaced_num)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path_root", type=str,
                        default='/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/dataset/')
    parser.add_argument("--dataset_folder", type=str, default='updated_gau_1000')
    args = parser.parse_args()
    print(args)

    dataset_folder = args.dataset_folder

    resize_dim = 112
    dataset_path_root = args.dataset_path_root
    dataset_path = os.path.join(dataset_path_root, dataset_folder)
    img_path = os.path.join(dataset_path, 'img')
    csv_path = os.path.join(dataset_path, 'labels.csv')
    label_dict = read_csv(csv_path)
    motor_difference = []
    # try to read labels_verify
    try:
        label_verify_path = os.path.join(dataset_path, 'labels_verify.csv')
        label_verify_dict = read_csv(label_verify_path)
    except:
        pass


    filenames = glob.glob(os.path.join(img_path, '*.png'))
    filenames.sort()


    for filename in tqdm(filenames):
        img_name = filename.split('/')[-1]

        number = img_name[4:-4]
        label = label_dict[img_name[4:-4]]
        try: # verify: if has -180 or too distant then continue
            label_verify = label_verify_dict[img_name[4:-4]]

            for ind in range(label.shape[0]):
                deg = label[ind]
                if abs(deg) > 180:
                    continue
            label_dist = abs(label_verify - label).mean()
            motor_difference.append(label_dist)
            if label_dist > 8:
                print('label distance too large (' + str(label_dist) + '), skipping ', img_name)
                continue
        except:
            pass


    # Creating a customized histogram with a density plot
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.histplot(motor_difference, bins=30, kde=True, color='lightgreen', edgecolor='red')

    # Adding labels and title
    plt.xlabel('Motor difference')
    plt.ylabel('Density')

    # Display the plot
    plt.show()