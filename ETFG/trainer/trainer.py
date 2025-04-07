import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dataset.dataset import MEADDataset, collate_fn, LandmarkNormalize, AudioNormalize
from model.model import MEADFeaturePredictor
from torchvision import transforms

'''if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"'''
device = "cpu"

def main():
    # Configuration of face_embed
    face_embed_config = {
        'feature_root': './MEAD_features/',
        'train_subjects': ['M003',],
        'val_subjects': ['M005'],
        'test_subjects': ['M005'],
        'emotion_labels': ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'neutral'],
        'feature_type': 'face_embed',  # 'ldmk' or 'face_embed'
        'batch_size': 32,
        'num_workers': 0,
        'max_epochs': 300,
        'learning_rate': 1e-4,
        'hidden_dim': 128,
        'num_layers': 3,
        'emotion_embedding_dim': 32,
        'log_name': 'face_embed',  # Name for the experiment logs
        'audio_transform': None,
        'target_transform': None
    }

    # Configuration of ldmk
    ldmk_config = {
        'feature_root': './MEAD_features/',
        'train_subjects': ['M003', ],
        'val_subjects': ['M005'],
        'test_subjects': ['M005'],
        'emotion_labels': ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'neutral'],
        'feature_type': 'ldmk',  # 'ldmk' or 'face_embed'
        'batch_size': 32,
        'num_workers': 0,
        'max_epochs': 300,
        'learning_rate': 1e-4,
        'hidden_dim': 64,
        'num_layers': 3,
        'emotion_embedding_dim': 32,
        'log_name': 'ldmk',  # Name for the experiment logs
        'audio_transform': None,
        'target_transform': transforms.Compose([
            LandmarkNormalize()
        ])
    }
    config = face_embed_config
    
    # Create datasets
    train_dataset = MEADDataset(
        root_dir=config['feature_root'],
        subjects=config['train_subjects'],
        emotion_labels=config['emotion_labels'],
        feature_type=config['feature_type'],
        audio_transform=config['audio_transform'],
        target_transform=config['target_transform']
    )
    
    val_dataset = MEADDataset(
        root_dir=config['feature_root'],
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
        feature_type=config['feature_type']
    )
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,  # Number of epochs with no improvement after which training will be stopped
        mode='min',
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename=f"{config['log_name']}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=2,  # Save only the best and latest checkpoints
        mode='min',
        save_last=True  # Always save the latest checkpoint
    )
    
    # Create logger
    logger = TensorBoardLogger(
        "lightning_logs",
        name=config['log_name']
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
    # todo: end token? assume same length?
    # todo: why max len is sometimes audio sometimes video?
    # audio to hubert feature.
    # audio transform no need!

    # input should be neutral sound?
    # output should be ldmk plus changed sound? -> not provided in the dataset, assume gpt generates
    # English only?