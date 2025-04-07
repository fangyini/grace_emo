import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MEADFeaturePredictor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 28,  # MFCC feature dimension
        hidden_dim: int = 256,
        num_layers: int = 3,
        output_dim: int = 136,  # 68 landmarks * 2 coordinates (flattened)
        learning_rate: float = 1e-4,
        emotion_embedding_dim: int = 32,
        num_emotions: int = 7,
        feature_type: str = 'ldmk',
        nhead: int = 8,
        dropout: float = 0.2,
        teacher_forcing_ratio: float = 0.5,
        architecture_type: str = 'encoder_only'  # 'encoder_only' or 'encoder_decoder'
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Emotion embedding layer
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embedding_dim)
        
        # Input projection layer
        self.input_projection_src = nn.Linear(input_dim + emotion_embedding_dim, hidden_dim)
        self.input_projection_tgt = nn.Linear(output_dim, hidden_dim)
        
        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer decoder layer (only used in encoder_decoder mode)
        if architecture_type == 'encoder_decoder':
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to predict landmarks
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Loss function
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        
    def forward(
        self, 
        x: torch.Tensor, 
        emotion_labels: torch.Tensor, 
        input_lengths: torch.Tensor,
        target_features: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False
    ) -> torch.Tensor:
        batch_size, input_seq_len, _ = x.size()
        
        # Process input features
        emotion_embeddings = self.emotion_embedding(emotion_labels)
        emotion_embeddings = emotion_embeddings.unsqueeze(1).expand(-1, input_seq_len, -1)
        combined_input = torch.cat([x, emotion_embeddings], dim=-1)
        
        x = self.input_projection_src(combined_input)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Create padding mask for encoder
        encoder_padding_mask = torch.arange(input_seq_len, device=x.device)[None, :] >= input_lengths[:, None]
        
        # Encode input features
        memory = self.transformer_encoder(x, src_key_padding_mask=encoder_padding_mask)
        
        if self.hparams.architecture_type == 'encoder_only':
            # In encoder-only mode, directly use the encoder output
            output = memory
        else:
            # Determine output sequence length
            if target_features is not None:
                output_seq_len = target_features.size(1)
            else:
                # If no target provided, use input length as default
                output_seq_len = input_seq_len
            
            # Initialize decoder input
            if target_features is not None and teacher_forcing:
                # Use target features as decoder input during teacher forcing
                tgt = self.input_projection_tgt(target_features)
            else:
                # Initialize with zeros
                tgt = torch.zeros(batch_size, output_seq_len, self.hparams.hidden_dim, device=x.device)
            
            tgt = self.pos_encoder(tgt.transpose(0, 1)).transpose(0, 1)
            
            # Create padding mask for decoder
            if target_lengths is not None:
                decoder_padding_mask = torch.arange(output_seq_len, device=x.device)[None, :] >= target_lengths[:, None]
            else:
                decoder_padding_mask = None
            
            # Decode to generate predictions
            output = self.transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_key_padding_mask=decoder_padding_mask,
                memory_key_padding_mask=encoder_padding_mask
            )
        
        # Generate predictions
        predictions = self.output_projection(output)
        
        return predictions
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        audio_features, target_features, emotion_labels, audio_lengths, target_lengths = batch
        
        # Determine if using teacher forcing for this step
        use_teacher_forcing = torch.rand(1).item() < self.hparams.teacher_forcing_ratio
        
        # Forward pass with teacher forcing
        predictions = self(
            audio_features, 
            emotion_labels, 
            audio_lengths,
            target_features=target_features if use_teacher_forcing else None,
            target_lengths=target_lengths,
            teacher_forcing=use_teacher_forcing
        )
        
        # Calculate loss only on non-padded positions
        mask = torch.arange(predictions.size(1), device=predictions.device)[None, :] < target_lengths[:, None]
        mask = mask.unsqueeze(-1)  # For face embeddings: [batch, seq_len, 512]
        mask = mask.expand_as(predictions)
        
        # Apply mask to predictions and targets
        masked_predictions = predictions * mask.float()
        masked_targets = target_features * mask.float()
        
        # Calculate loss
        loss = self.MSE(masked_predictions, masked_targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        audio_features, target_features, emotion_labels, audio_lengths, target_lengths = batch
        
        # During validation, no teacher forcing
        predictions = self(
            audio_features, 
            emotion_labels, 
            audio_lengths,
            target_features=None,
            target_lengths=target_lengths,
            teacher_forcing=False
        )
        
        # Calculate loss only on non-padded positions
        mask = torch.arange(predictions.size(1), device=predictions.device)[None, :] < target_lengths[:, None]
        mask = mask.unsqueeze(-1)  # For face embeddings: [batch, seq_len, 512]
        mask = mask.expand_as(predictions)
        
        # Apply mask to predictions and targets
        masked_predictions = predictions * mask.float()
        masked_targets = target_features * mask.float()
        
        # Calculate loss
        loss = self.MAE(masked_predictions, masked_targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss}
    
    def test_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        audio_features, target_features, emotion_labels, audio_lengths, target_lengths = batch
        
        # During testing, no teacher forcing
        predictions = self(
            audio_features, 
            emotion_labels, 
            audio_lengths,
            target_features=None,
            target_lengths=target_lengths,
            teacher_forcing=False
        )
        
        # Calculate loss only on non-padded positions
        mask = torch.arange(predictions.size(1), device=predictions.device)[None, :] < target_lengths[:, None]
        mask = mask.unsqueeze(-1)  # For face embeddings: [batch, seq_len, 512]
        mask = mask.expand_as(predictions)
        
        # Apply mask to predictions and targets
        masked_predictions = predictions * mask.float()
        masked_targets = target_features * mask.float()
        
        # Calculate loss
        loss = self.MAE(masked_predictions, masked_targets)
        
        # Log test metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'test_loss': loss}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer 