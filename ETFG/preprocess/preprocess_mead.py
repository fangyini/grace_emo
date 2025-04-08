# for each input: video+audio,
# convert to MFCC feature, landmarks and facial features, save the file and path in csv,
# be able to skip, maintain the same file structures, subject/MFCC/label/level, subject/facial_feat, subject_ldmk,
import os
import numpy as np
from extract_audio import extract_mfcc
from extract_facial import extract_facial, get_num_of_frames
from tqdm import tqdm
import argparse

# Unified feature extraction for facial features and landmarks, and save together with MFCC in .npz
def extract_all_features(detection_bs, feat_bs, feature_choice, audio_path=None, video_path=None):
    features = {}
    if feature_choice == 'visual':
        ldmk, face_embed = extract_facial(video_path, detection_bs, feat_bs)
        features['face_embed'] = face_embed
        features['ldmk'] = ldmk
    elif feature_choice == 'audio':
        frame_num = get_num_of_frames(video_path)
        try:
            mfcc_output = extract_mfcc(audio_path, target_num_frames=frame_num)
            features['mfcc'] = mfcc_output
        except Exception as e:
            return None
    #assert mfcc_output.shape[0] == face_embed.shape[0]
    return features

def process_all_modalities(input_root, output_root, level, camera_degree, partition, total_partition, detection_bs, feat_bs, feature_choice, audio_from_video=False):
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

        if not os.path.exists(video_base):
            continue

        # Get emotion labels from video directory
        emotion_labels = os.listdir(video_base)
        emotion_labels = [i for i in emotion_labels if i != '.DS_Store']
        
        for emotion_label in tqdm(emotion_labels):
            # Determine which level to use for neutral emotion
            if emotion_label == 'neutral' and level != 'level_1':
                video_level_path = os.path.join(video_base, emotion_label, 'level_1')
                audio_level_path = os.path.join(audio_base, emotion_label, 'level_1') if os.path.exists(audio_base) else None
            else:
                video_level_path = os.path.join(video_base, emotion_label, level)
                audio_level_path = os.path.join(audio_base, emotion_label, level) if os.path.exists(audio_base) else None
                
            if not os.path.exists(video_level_path):
                continue

            video_files = sorted(os.listdir(video_level_path))
            
            # Get files to process based on feature_choice
            if feature_choice == 'visual':
                # For visual features, process all video files
                process_files = video_files
            else:  # feature_choice == 'audio'
                # For audio features, only process files that have both audio and video
                if not audio_level_path or not os.path.exists(audio_level_path):
                    continue
                audio_files = sorted(os.listdir(audio_level_path))
                # Create a set of available audio files without extension
                available_audio = {os.path.splitext(f)[0] for f in audio_files}
                # Only process video files that have corresponding audio files
                process_files = [f for f in video_files if os.path.splitext(f)[0] in available_audio]

            if not process_files:
                continue

            # For non-neutral emotions, use the specified level
            # For neutral emotion, use level_1 regardless of the specified level
            output_level = 'level_1' if emotion_label == 'neutral' else level
            output_path = os.path.join(output_root, subject, emotion_label, output_level)
            os.makedirs(output_path, exist_ok=True)

            # Skip if already processed
            if len(os.listdir(output_path)) == len(process_files):
                print(f"Skipping {output_path} - all files already processed")
                continue

            for video_file in process_files:
                base_name = os.path.splitext(video_file)[0]
                out_file = os.path.join(output_path, base_name + '.npz')

                if os.path.exists(out_file):
                    print(f"Skipping {out_file} - feature file already exists")
                    continue
                    
                video_path = os.path.join(video_level_path, video_file)
                
                # Set audio path based on feature_choice and audio_from_video
                if feature_choice == 'visual':
                    # For visual features, audio path is not needed
                    audio_path = None
                else:  # feature_choice == 'audio'
                    if audio_from_video:
                        # Use video file for audio extraction
                        audio_path = video_path
                    else:
                        # Use audio file
                        audio_file = base_name + '.m4a'
                        audio_path = os.path.join(audio_level_path, audio_file)
                        if not os.path.exists(audio_path):
                            continue

                features = extract_all_features(detection_bs, feat_bs, audio_path=audio_path, video_path=video_path, feature_choice=feature_choice)
                if features is not None:
                    np.savez(out_file, **features)

parser = argparse.ArgumentParser(description='Dataset process')
parser.add_argument('--root_dir', type=str, default='/Users/xiaokeai/Documents/HKUST/datasets/MEAD/data/')
parser.add_argument('--feature_path', type=str, default='../MEAD_audio_from_video_features/')
parser.add_argument('--process_level', type=str, default='level_2')
parser.add_argument('--camera_degree', type=str, default='front')
parser.add_argument('--feature_choice', type=str, default='audio') # audio, visual
parser.add_argument('--audio_from_video', action='store_true')

parser.add_argument('--partition', type=int, default=0)
parser.add_argument('--total_partition', type=int, default=1)

parser.add_argument('--detection_bs', type=int, default=16)
parser.add_argument('--feat_bs', type=int, default=64)

args = parser.parse_args()
print(args)
root_dir = args.root_dir
feature_path = args.feature_path
process_level = args.process_level
camera_degree = args.camera_degree

os.makedirs(feature_path, exist_ok=True)
process_all_modalities(root_dir, feature_path, level=process_level, camera_degree=camera_degree,
                       partition=args.partition, total_partition=args.total_partition,
                       detection_bs=args.detection_bs, feat_bs=args.feat_bs, feature_choice=args.feature_choice,
                       audio_from_video=args.audio_from_video)
