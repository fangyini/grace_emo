import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from ETFG.dataset.dataset import MEADDataset, collate_fn, LandmarkNormalize, AudioNormalize
from ETFG.model.model import MEADFeaturePredictor
from torchvision import transforms
from ETFG.config import get_config

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main():
    # Get configuration from YAML file
    config = get_config()
    
    # Set up transforms
    if config['feature_type'] == 'ldmk':
        config['target_transform'] = transforms.Compose([
            LandmarkNormalize()
        ])
    
    # Create datasets
    train_dataset = MEADDataset(
        visual_root_dir=config['visual_feature_root'],
        audio_root_dir=config['audio_feature_root'],
        subjects=config['train_subjects'],
        emotion_labels=config['emotion_labels'],
        feature_type=config['feature_type'],
        audio_transform=config['audio_transform'],
        target_transform=config['target_transform']
    )
    
    val_dataset = MEADDataset(
        visual_root_dir=config['visual_feature_root'],
        audio_root_dir=config['audio_feature_root'],
        subjects=config['val_subjects'],
        emotion_labels=config['emotion_labels'],
        feature_type=config['feature_type'],
        audio_transform=config['audio_transform'],
        target_transform=config['target_transform']
    )
    
    # Create dataloaders with collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create model
    output_dim = 512 if config['feature_type'] == 'face_embed' else 136  # 512 for face_embed, 136 for landmarks (68*2)
    model = MEADFeaturePredictor(
        input_dim=28,  # MFCC feature dimension
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        output_dim=output_dim,
        learning_rate=config['learning_rate'],
        emotion_embedding_dim=config['emotion_embedding_dim'],
        num_emotions=len(config['emotion_labels']),
        feature_type=config['feature_type'],
        architecture_type=config['architecture_type']
    )
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,  # Number of epochs with no improvement after which training will be stopped
        mode='min',
        verbose=True
    )
    
    # Create logger
    logger = TensorBoardLogger(
        config['lightning_log_root'],
        name=config['log_name']
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename=f"{config['log_name']}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=2,  # Save only the best and latest checkpoints
        mode='min',
        save_last=True  # Always save the latest checkpoint
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator=device,
        devices=1,
        precision=16 if torch.cuda.is_available() else 32
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    '''test_dataset = MEADDataset(
        root_dir=config['feature_root'],
        subjects=config['test_subjects'],
        emotion_labels=config['emotion_labels'],
        feature_type=config['feature_type'],
        audio_transform=config['audio_transform'],
        target_transform=config['target_transform']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    trainer.test(model, test_loader)'''

if __name__ == '__main__':
    main()
