import csv
import numpy as np
import matplotlib.pyplot as plt

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

testing_output = read_csv('grace_emo/emotion/motor_prediction/lightning_logs/5-layer NN_lrelu/testing_results.csv')
gt = read_csv('grace_emo/dataset/gau_600/labels.csv')
all_data = []
diff_per_trail = []
titles = []
for x in testing_output:
    gt_labels = gt[x]
    predicted = testing_output[x]
    diff = abs(gt_labels - predicted).tolist()
    all_data.extend(diff)
    diff_per_trail.append(np.mean(diff))
    titles.append(x)

lessthanone = np.where(np.stack(all_data) < 1)[0]
print(len(lessthanone)/len(all_data))
diff_per_trail = np.stack(diff_per_trail)
args = np.argsort(diff_per_trail)[::-1]
print(diff_per_trail[args[0]], titles[args[0]])
print(diff_per_trail[args[1]], titles[args[1]])
print(diff_per_trail[args[2]], titles[args[2]])
quit()
plt.hist(all_data, bins=50)
plt.show()
