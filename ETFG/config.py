import argparse
import yaml
from typing import Dict, Any
import os

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file
    Args:
        config_path: Path to the YAML configuration file
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add any additional configuration that doesn't come from YAML
    config['audio_transform'] = None
    config['target_transform'] = None
    
    return config

def get_config() -> Dict[str, Any]:
    """
    Get configuration by loading from YAML file specified in command line arguments
    """
    parser = argparse.ArgumentParser(description='MEAD Feature Predictor Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file (e.g., configs/face_embed.yaml or configs/ldmk.yaml)')
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    return config
