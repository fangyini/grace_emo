# images: face alignment, create cleaned dataset, save landmark, remove not moving images
# check repeated images in csv, create cleaned label
# check motors state
# check gaussian

import os
import glob
import cv2
import dlib
from tqdm import tqdm
import csv
import numpy as np
from local_test.calculate_sim import shape_to_np, rect_to_bb
from pytorch_face_landmark.test_batch_detections import get_face_and_ldmk, load_model

def detect_face(image, getBoundingBox=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = detector(gray, 1)[0]
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)

    (x, y, w, h) = rect_to_bb(rect)
    if getBoundingBox:
        return (x, y, w, h)
    '''cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Face #{}".format(1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    return shape

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
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('pretrained/shape_predictor_68_face_landmarks.dat')
    face_model = load_model()

    resize_dim = 112
    dataset_path_root = '/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/dataset/'
    dataset_path = os.path.join(dataset_path_root, 'gau_600')
    img_path = os.path.join(dataset_path, 'img')
    csv_path = os.path.join(dataset_path, 'labels.csv')
    landmark_path = os.path.join(dataset_path, 'landmarks.csv')
    label_dict = read_csv(csv_path)
    # try to read labels_verify
    try:
        label_verify_path = os.path.join(dataset_path, 'labels_verify.csv')
        label_verify_dict = read_csv(label_verify_path)
    except:
        pass

    processed_dataset_path = os.path.join(dataset_path_root, 'processed_gau_600')
    if not os.path.exists(processed_dataset_path):
        os.makedirs(processed_dataset_path)
    processed_img_path = os.path.join(processed_dataset_path, 'data')
    if not os.path.exists(processed_img_path):
        os.makedirs(processed_img_path)

    filenames = glob.glob(os.path.join(img_path, '*.png'))
    filenames.sort()

    neutral = os.path.join(dataset_path, 'img_neutral.png')
    neutral = cv2.imread(neutral)

    #(x, y, w, h) = detect_face(neutral, getBoundingBox=True)
    (x, y, w, h) = get_face_and_ldmk(neutral, face_model, getBoundingBox=True)
    # check last landmark, if too close then continue the loop

    cropped_image = neutral[y:y + h, x:x + w]
    cropped_image = cv2.resize(cropped_image, (resize_dim, resize_dim))

    #last_landmark = detect_face(cropped_image) # nparray, 68,2
    last_landmark = get_face_and_ldmk(cropped_image, face_model)

    for filename in tqdm(filenames):
        img_name = filename.split('/')[-1]
        image = cv2.imread(filename)
        cropped_image = image[y:y + h, x:x + w]
        cropped_image = cv2.resize(cropped_image, (resize_dim, resize_dim))

        #landmark = detect_face(cropped_image)
        landmark = get_face_and_ldmk(cropped_image, face_model)

        ldmk_distance = abs(landmark - last_landmark).sum()
        #print(img_name, ldmk_distance)
        if ldmk_distance < 100:
            print('landmark distance too small (' +str(ldmk_distance)+ '), skipping ', img_name)
            continue
        number = img_name[4:-4]
        label = label_dict[img_name[4:-4]]
        try: # verify: if has -180 or too distant then continue
            label_verify = label_verify_dict[img_name[4:-4]]
            for ind in range(label.shape[0]):
                deg = label[ind]
                if abs(deg) > 180:
                    continue
            label_dist = abs(label_verify - label).sum()
            if label_dist > 100:
                print('label distance too large (' + str(label_dist) + '), skipping ', img_name)
                continue
        except:
            pass

        # save npz, image(224,224,3)/landmark(68,2)/label(26,)
        np.savez(os.path.join(processed_img_path, 'data_'+number+'.npz'), image=cropped_image, label=label, ldmk=landmark)
        last_landmark = landmark

        # filters: motor errors, degrees>180, landmarks distance, label distance
    # original num vs. processed num?