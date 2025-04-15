import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Collate function to handle variable length sequences with padding
    Args:
        batch: List of tuples (audio_features, target_features, emotion_label)
    Returns:
        Tuple of padded tensors and sequence lengths
    """
    # Unpack the batch
    audio_features, target_features, emotion_labels = zip(*batch)
    
    # Get sequence lengths
    audio_lengths = torch.tensor([len(x) for x in audio_features])
    target_lengths = torch.tensor([len(x) for x in target_features])
    
    # Pad sequences to the maximum length in the batch
    audio_padded = pad_sequence(audio_features, batch_first=True, padding_value=0.0) # bs,143,28
    target_padded = pad_sequence(target_features, batch_first=True, padding_value=0.0) # bs,145,512
    
    # Stack emotion labels
    emotion_labels = torch.stack(emotion_labels)
    
    return audio_padded, target_padded, emotion_labels, audio_lengths, target_lengths

class LandmarkNormalize:
    """Normalize landmarks by dividing by 112"""
    def __call__(self, x):
        return x / 112.0

class AudioNormalize:
    """Normalize audio features using Z-score normalization"""
    def __call__(self, x):
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        # Add small epsilon to avoid division by zero
        std = torch.clamp(std, min=1e-8)
        return (x - mean) / std

class MEADDataset(Dataset):
    def __init__(self, 
                 visual_root_dir: str,
                 audio_root_dir: str,
                 subjects: List[str],
                 emotion_labels: List[str],
                 intensity_level,
                 feature_type: str = 'ldmk',  # 'ldmk' or 'face_embed'
                 audio_transform=None,
                 target_transform=None
                 ):
        """
        Args:
            visual_root_dir: Path to the directory containing visual features (ldmk and face_embed)
            audio_root_dir: Path to the directory containing audio features (mfcc)
            subjects: List of subject IDs to include
            emotion_labels: List of emotion labels to include
            feature_type: Type of feature to use as target ('ldmk' or 'face_embed')
            transform: Optional transforms to apply
        """
        self.visual_root_dir = visual_root_dir
        self.audio_root_dir = audio_root_dir
        self.subjects = subjects
        self.emotion_labels = emotion_labels
        self.feature_type = feature_type
        self.audio_transform = audio_transform
        self.target_transform = target_transform
        
        # Create emotion to index mapping
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
        
        # Collect all feature files
        self.feature_files = []
        self.emotion_indices = []
        
        # Keep track of skipped files for reporting
        skipped_files = {
            'missing_subject': 0,
            'missing_emotion': 0,
            'missing_level': 0,
            'missing_audio': 0
        }
        
        for subject in subjects:
            # Visual features path
            visual_subject_path = os.path.join(visual_root_dir, subject)
            # Audio features path
            audio_subject_path = os.path.join(audio_root_dir, subject)
            
            if not os.path.exists(visual_subject_path) or not os.path.exists(audio_subject_path):
                print(f"Warning: Subject {subject} not found in either visual or audio directory")
                skipped_files['missing_subject'] += 1
                continue
                
            for emotion in emotion_labels:
                visual_emotion_path = os.path.join(visual_subject_path, emotion)
                audio_emotion_path = os.path.join(audio_subject_path, emotion)
                
                if not os.path.exists(visual_emotion_path) or not os.path.exists(audio_emotion_path):
                    print(f"Warning: Emotion {emotion} not found for subject {subject}")
                    skipped_files['missing_emotion'] += 1
                    continue
                    
                # Look for level_1 for neutral, level_2 for others
                level = 'level_1' if emotion == 'neutral' else intensity_level
                visual_level_path = os.path.join(visual_emotion_path, level)
                audio_level_path = os.path.join(audio_emotion_path, level)
                
                if not os.path.exists(visual_level_path) or not os.path.exists(audio_level_path):
                    print(f"Warning: Level {level} not found for {emotion} in subject {subject}")
                    skipped_files['missing_level'] += 1
                    continue
                
                # Get all visual feature files
                visual_files = [f for f in os.listdir(visual_level_path) if f.endswith('.npz')]
                for visual_file in visual_files:
                    # Check if corresponding audio file exists
                    audio_file = os.path.join(audio_level_path, visual_file)
                    if not os.path.exists(audio_file):
                        skipped_files['missing_audio'] += 1
                        continue
                        
                    # Both files exist, add them to the dataset
                    self.feature_files.append((
                        os.path.join(visual_level_path, visual_file),
                        audio_file
                    ))
                    self.emotion_indices.append(self.emotion_to_idx[emotion])
        
        # Print summary of loaded and skipped files
        print(f"\nDataset Loading Summary:")
        print(f"Successfully loaded {len(self.feature_files)} video pairs")
        print(f"Skipped due to missing subject: {skipped_files['missing_subject']}")
        print(f"Skipped due to missing emotion: {skipped_files['missing_emotion']}")
        print(f"Skipped due to missing level: {skipped_files['missing_level']}")
        print(f"Skipped due to missing audio file: {skipped_files['missing_audio']}")
        
    def __len__(self) -> int:
        return len(self.feature_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset
        Args:
            idx: Index of the sample
        Returns:
            Tuple of (audio_features, target_features, emotion_label)
        """
        # Get the feature file paths and emotion index
        visual_file, audio_file = self.feature_files[idx]
        emotion_idx = self.emotion_indices[idx]
        
        # Load visual features
        visual_features = np.load(visual_file)
        target_features = torch.from_numpy(visual_features[self.feature_type]).float()
        if self.feature_type == 'ldmk':
            target_features = torch.flatten(target_features, 1, 2)
        # Load audio features
        audio_features = np.load(audio_file)
        audio_features = torch.from_numpy(audio_features['mfcc']).float()
        
        # Apply transforms if specified
        if self.audio_transform:
            audio_features = self.audio_transform(audio_features)
        if self.target_transform:
            target_features = self.target_transform(target_features)
            
        # Convert emotion index to tensor
        emotion_label = torch.tensor(emotion_idx, dtype=torch.long)
        
        return audio_features, target_features, emotion_label 