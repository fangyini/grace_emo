import os
import torch
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
from ETFG.model.model import MEADFeaturePredictor
from emotion.motor_prediction.model import GraceModel
from emotion.motor_prediction.trainer import GracePL
from emotion.motor_prediction.utils import calculate_data_stat
import pytorch_lightning as pl
from ETFG.dataset.dataset import MEADDataset, collate_fn
from torch.utils.data import DataLoader
import yaml
import argparse
import csv
def load_config(config_path: str) -> Dict:
    """
    Load configuration from a YAML file
    Args:
        config_path: Path to the YAML configuration file
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class EvaluationPipeline:
    def __init__(
        self,
        config: Dict
    ):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.feature_type = config['feature_type']
        self.save_path = config['save_path']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load ETFG model
        self.etfg_model = MEADFeaturePredictor.load_from_checkpoint(
            config['etfg_checkpoint'],
            feature_type=self.feature_type
        ).to(self.device)
        self.etfg_model.eval()
        
        # Calculate data statistics
        self.label_mean, self.label_std, _, _ = calculate_data_stat(config['dataset']['grace_data_root_dir'])

        # Load motor prediction model
        # First load the Lightning model
        motor_pl = GracePL.load_from_checkpoint(
            config['motor_checkpoint'],
            output_path='',  # Not used in evaluation
            label_mean=self.label_mean,
            label_std=self.label_std,
            learning_rate=1e-4,  # Dummy value, not used in evaluation
            feature_type=self.feature_type
        )
        # Get the actual model
        self.motor_model = motor_pl.model.to(self.device)
        self.motor_model.eval()
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
    
    def load_test_data(self) -> Tuple[DataLoader, List[str]]:
        """
        Load test data using MEADDataset.
        
        Returns:
            Tuple of (test_loader, video_names)
            test_loader: DataLoader for the test dataset
            video_names: List of video names
        """
        dataset_config = self.config['dataset']
        
        # Create dataset
        test_dataset = MEADDataset(
            visual_root_dir=dataset_config['visual_root_dir'],
            audio_root_dir=dataset_config['audio_root_dir'],
            subjects=dataset_config['subjects'],
            emotion_labels=dataset_config['emotion_labels'],
            intensity_level=dataset_config['intensity_level'],
            feature_type=self.feature_type
        )
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one video at a time
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Get video names from feature files
        video_names = [f[0].replace('.npz', '').split('/')[-4:] for f in test_dataset.feature_files]
        
        return test_loader, video_names
    
    def process_video(self, video_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process a single video through the pipeline.
        
        Args:
            video_data: Dictionary containing input features and emotion labels
            
        Returns:
            Predicted motor commands for each frame
        """
        with torch.no_grad():
            # Get features from ETFG model (sequence of features)
            features = self.etfg_model(
                video_data['input_features'],
                video_data['emotion_labels'],
                video_data['input_lengths']
            )
            
            # Process all frames through the motor prediction model in batch
            # features shape: [seq_len, feature_dim]
            # Add batch dimension: [1, seq_len, feature_dim]
            features = features.squeeze(0)
            # Get motor commands for all frames in batch
            motor_commands = self.motor_model(features)

            # Denormalize the motor commands
            motor_commands = motor_commands * self.label_std + self.label_mean
            
            return motor_commands
    
    def evaluate(self, test_loader: DataLoader, video_names: List[str]):
        """
        Evaluate the pipeline on test data.
        
        Args:
            test_loader: DataLoader for the test dataset
            video_names: List of video names corresponding to the test data
        """
        results = []
        
        for batch_idx, (audio_features, target_features, emotion_labels, audio_lengths, target_lengths) in enumerate(test_loader):
            video_name = video_names[batch_idx]
            video_name = "/".join(video_name)
            # Move data to device
            audio_features = audio_features.to(self.device)
            emotion_labels = emotion_labels.to(self.device)
            audio_lengths = audio_lengths.to(self.device)
            
            # Process video
            video_data = {
                'input_features': audio_features,
                'emotion_labels': emotion_labels,
                'input_lengths': audio_lengths
            }
            motor_commands = self.process_video(video_data)
            
            # Convert to numpy and save
            motor_commands_np = motor_commands.cpu().numpy() # n, 26
            video_name_frames = [video_name+'/'+str(x).zfill(3) for x in range(motor_commands_np.shape[0])]
            for y in range(motor_commands_np.shape[0]):
                results.append([video_name_frames[y], motor_commands_np[y].tolist()])
        
        # Save results
        output_file = os.path.join(self.save_path, 'motor_commands.txt')
        with open(output_file, 'w') as csv_writer:
            writer = csv.writer(csv_writer, delimiter='\t',lineterminator='\n',)
            for result in results:
                writer.writerow(result)

        print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate the ETFG and motor prediction pipeline')
    parser.add_argument('--config', type=str, default='configs/face_embed.yaml')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(config)
    
    # Load test data
    test_loader, video_names = pipeline.load_test_data()
    
    # Evaluate
    pipeline.evaluate(test_loader, video_names)

if __name__ == '__main__':
    main()