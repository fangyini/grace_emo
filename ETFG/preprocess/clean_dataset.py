import os
import tarfile
import shutil
import argparse
from pathlib import Path

def process_subject(subject_path):
    # Unzip audio.tar
    audio_tar = subject_path / "audio.tar"
    if audio_tar.exists():
        with tarfile.open(audio_tar, "r") as tar:
            tar.extractall(path=subject_path)
        os.remove(audio_tar)
    
    # Unzip video.tar
    video_tar = subject_path / "video.tar"
    if video_tar.exists():
        with tarfile.open(video_tar, "r") as tar:
            tar.extractall(path=subject_path)
        os.remove(video_tar)
    
        # Remove specified video folders
        video_path = subject_path / "video"
        if video_path.exists():
            folders_to_remove = ["down", "left_30", "left_60", "right_30", "right_60", "top"]
            for folder in folders_to_remove:
                folder_path = video_path / folder
                if folder_path.exists():
                    shutil.rmtree(folder_path)

def main():
    # Convert root directory to Path object
    root_dir = Path(args.root_dir)
    
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist")
        return
    
    # Process each subject directory
    for subject_dir in root_dir.iterdir():
        if subject_dir.is_dir():
            print(f"Processing {subject_dir.name}...")
            process_subject(subject_dir)
            print(f"Completed processing {subject_dir.name}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process subject directories containing audio and video data.')
    parser.add_argument('--root_dir', default='/Users/xiaokeai/Downloads/MEAD/', type=str)
    args = parser.parse_args()
    main()
