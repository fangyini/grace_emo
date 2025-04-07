# for each input: video+audio,
# convert to MFCC feature, landmarks and facial features, save the file and path in csv,
# be able to skip, maintain the same file structures, subject/MFCC/label/level, subject/facial_feat, subject_ldmk,
import os
import numpy as np
from extract_audio import extract_mfcc
from extract_facial import extract_facial
from tqdm import tqdm
import argparse

# Unified feature extraction for facial features and landmarks, and save together with MFCC in .npz
def extract_all_features(audio_path=None, video_path=None):
    ldmk, face_embed = extract_facial(video_path)
    mfcc_output = extract_mfcc(audio_path, target_num_frames=face_embed.shape[0])
    assert mfcc_output.shape[0] == face_embed.shape[0]
    features = {}
    features['mfcc'] = mfcc_output
    features['face_embed'] = face_embed
    features['ldmk'] = ldmk
    return features

def process_all_modalities(input_root, output_root, level, camera_degree, partition, total_partition):
    subject = os.listdir(input_root)
    subject = [i for i in subject if i != '.DS_Store']
    subject.sort()

    total_number = len(subject)
    every_part = round(total_number / total_partition)
    process_subjects = subject[(every_part * partition):(every_part * partition + every_part)]

    for subject in process_subjects:
        print(subject)

        subject_input = os.path.join(input_root, subject)
        if not os.path.isdir(subject_input):
            continue

        audio_base = os.path.join(subject_input, 'audio')
        video_base = os.path.join(subject_input, 'video', camera_degree)

        if not os.path.exists(audio_base) or not os.path.exists(video_base):
            continue

        # Find overlapping emotion_labels
        emotion_labels = list(set(os.listdir(audio_base)).intersection(os.listdir(video_base)))
        emotion_labels = [i for i in emotion_labels if i != '.DS_Store']
        
        for emotion_label in tqdm(emotion_labels):
            # Determine which level to use for neutral emotion
            if emotion_label == 'neutral' and level != 'level_1':
                audio_level_path = os.path.join(audio_base, emotion_label, 'level_1')
                video_level_path = os.path.join(video_base, emotion_label, 'level_1')
            else:
                audio_level_path = os.path.join(audio_base, emotion_label, level)
                video_level_path = os.path.join(video_base, emotion_label, level)
                
            if not os.path.exists(audio_level_path) or not os.path.exists(video_level_path):
                continue

            audio_files = sorted(os.listdir(audio_level_path))
            video_files = sorted(os.listdir(video_level_path))

            # Check file count match
            if len(audio_files) != len(video_files):
                print('WARNING! The audio file count is different than video file count! ' + audio_level_path + ' and ' + video_level_path)
                continue

            # For non-neutral emotions, use the specified level
            # For neutral emotion, use level_1 regardless of the specified level
            output_level = 'level_1' if emotion_label == 'neutral' else level
            output_path = os.path.join(output_root, subject, emotion_label, output_level)
            os.makedirs(output_path, exist_ok=True)

            # Skip if already processed
            if len(os.listdir(output_path)) == len(audio_files):
                print(f"Skipping {output_path} - all files already processed")
                continue

            for audio_file, video_file in zip(audio_files, video_files):
                base_name = os.path.splitext(audio_file)[0]
                out_file = os.path.join(output_path, base_name + '.npz')

                if os.path.exists(out_file):
                    print(f"Skipping {out_file} - feature file already exists")
                    continue
                    
                audio_path = os.path.join(audio_level_path, audio_file)
                video_path = os.path.join(video_level_path, video_file)

                features = extract_all_features(audio_path=audio_path, video_path=video_path)
                np.savez(out_file, **features)
                #if subject == 'M005':
                #    break

parser = argparse.ArgumentParser(description='Dataset process')
parser.add_argument('--root_dir', type=str, default='/Users/xiaokeai/Documents/HKUST/datasets/MEAD/')
parser.add_argument('--feature_path', type=str, default='../MEAD_features/')
parser.add_argument('--process_level', type=str, default='level_2')
parser.add_argument('--camera_degree', type=str, default='front')

parser.add_argument('--partition', type=int, default=0)
parser.add_argument('--total_partition', type=int, default=1)


args = parser.parse_args()

root_dir = args.root_dir
feature_path = args.feature_path
process_level = args.process_level
camera_degree = args.camera_degree



os.makedirs(feature_path, exist_ok=True)
process_all_modalities(root_dir, feature_path, level=process_level, camera_degree=camera_degree,
                       partition=args.partition, total_partition=args.total_partition)
