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
                 root_dir: str,
                 subjects: List[str],
                 emotion_labels: List[str],
                 feature_type: str = 'ldmk',  # 'ldmk' or 'face_embed'
                 audio_transform=None,
                 target_transform=None
                 ):
        """
        Args:
            root_dir: Path to the MEAD features directory
            subjects: List of subject IDs to include
            emotion_labels: List of emotion labels to include
            feature_type: Type of feature to use as target ('ldmk' or 'face_embed')
            transform: Optional transforms to apply
        """
        self.root_dir = root_dir
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
        print('testing code, delete later.')
        
        for subject in subjects:
            subject_path = os.path.join(root_dir, subject)
            if not os.path.exists(subject_path):
                print(f"Warning: Subject {subject} not found")
                continue
                
            for emotion in emotion_labels:
                emotion_path = os.path.join(subject_path, emotion)
                if not os.path.exists(emotion_path):
                    print(f"Warning: Emotion {emotion} not found for subject {subject}")
                    continue
                    
                # Look for level_1 for neutral, level_2 for others
                level = 'level_1' if emotion == 'neutral' else 'level_2'
                level_path = os.path.join(emotion_path, level)
                
                if not os.path.exists(level_path):
                    print(f"Warning: Level {level} not found for {emotion} in subject {subject}")
                    continue
                
                # Add all npz files in this directory
                for file in os.listdir(level_path):
                    if file.endswith('.npz'):
                        self.feature_files.append(os.path.join(level_path, file))
                        self.emotion_indices.append(self.emotion_to_idx[emotion])
        
        print(f"Loaded {len(self.feature_files)} feature files")
        
    def __len__(self) -> int:
        return len(self.feature_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (audio_features, target_features, emotion_label)
        """
        # Load features from npz file
        features = np.load(self.feature_files[idx])
        
        # Extract features
        audio_features = torch.FloatTensor(features['mfcc'])
        target_features = torch.FloatTensor(features[self.feature_type])
        if self.feature_type == 'ldmk':
            target_features = target_features.flatten(1, 2)
        emotion_label = torch.LongTensor([self.emotion_indices[idx]])[0]
        # todo testing only, delete later
        target_features = target_features[:audio_features.size()[0]]
        # Apply transforms if any
        if self.audio_transform:
            audio_features = self.audio_transform(audio_features)
        if self.target_transform:
            target_features = self.target_transform(target_features)
        
        return audio_features, target_features, emotion_label 