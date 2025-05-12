import numpy as np
import os
import cv2
from pathlib import Path
import argparse

def extract_frame_from_video(video_path, frame_number):
    """Extract a specific frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from video: {video_path}")
    
    return frame

def process_directory(test_dir, gt_dir, output_dir):
    """Process all images in the test directory and save corresponding ground truth frames."""
    test_path = Path(test_dir)
    gt_path = Path(gt_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Walk through the test directory structure
    for subject_dir in test_path.iterdir():
        if not subject_dir.is_dir():
            continue
            
        for emotion_dir in subject_dir.iterdir():
            if not emotion_dir.is_dir():
                continue
                
            for level_dir in emotion_dir.iterdir():
                if not level_dir.is_dir():
                    continue
                    
                for video_dir in level_dir.iterdir():
                    if not video_dir.is_dir():
                        continue
                        
                    # Construct corresponding ground truth video path
                    subject = subject_dir.name
                    emotion = emotion_dir.name
                    level = level_dir.name
                    video_num = video_dir.name
                    
                    gt_video_path = gt_path / subject / "video" / "front" / emotion / level / f"{video_num}.mp4"
                    
                    if not gt_video_path.exists():
                        print(f"Warning: Ground truth video not found: {gt_video_path}")
                        continue
                    
                    # Process each image in the video directory
                    for img_file in video_dir.glob("*.png"):
                        # Extract frame number from image filename (assuming format like "frame_0001.png")
                        try:
                            frame_num = int(img_file.stem.split('.')[0])
                        except (IndexError, ValueError):
                            print(f"Warning: Could not parse frame number from {img_file}")
                            continue
                        
                        try:
                            # Extract frame from ground truth video
                            frame = extract_frame_from_video(str(gt_video_path), frame_num)
                            
                            # Create output path maintaining the same directory structure
                            rel_path = img_file.relative_to(test_path)
                            output_img_path = output_path / rel_path
                            output_img_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Save the frame
                            cv2.imwrite(str(output_img_path), frame)
                            print(f"Saved ground truth frame: {output_img_path}")
                            
                        except Exception as e:
                            print(f"Error processing {img_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Generate ground truth images from videos')
    parser.add_argument('--test_dir', type=str, default='/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/dataset/generated_photos_face_embed/img/')
    parser.add_argument('--gt_dir', type=str, default='/Users/xiaokeai/Documents/HKUST/datasets/MEAD/data')
    parser.add_argument('--output_dir', type=str, default='/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/dataset/generated_photos_face_embed/gt_img/')
    
    args = parser.parse_args()
    
    process_directory(args.test_dir, args.gt_dir, args.output_dir)

if __name__ == "__main__":
    main()