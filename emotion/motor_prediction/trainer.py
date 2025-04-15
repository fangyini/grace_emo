import os
from torch import optim, nn
import lightning as L
from emotion.motor_prediction.dataset import GraceFaceDataset
from emotion.motor_prediction.model import GraceModel
from emotion.motor_prediction.utils import get_device
from torch import optim, nn, utils, Tensor
from emotion.motor_prediction.utils import calculate_data_stat
from torch.utils.data import DataLoader
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import argparse
import pandas as pd
import csv
import torch
device = get_device()

# define the LightningModule
class GracePL(L.LightningModule):
    def __init__(self, output_path, label_mean, label_std,
                 learning_rate, feature_type='face_embed', autoML=False):
        super().__init__()
        self.model = GraceModel(feature_type)
        self.mse_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        self.label_mean = label_mean
        self.label_std = label_std
        self.lr = learning_rate
        self.feature_type = feature_type
        self.isAutoML = autoML
        print('Output path=', output_path)
        self.csv_file = os.path.join(output_path, 'testing_results.csv')
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        
        # Labels are still normalized with z-score
        labels = (labels - self.label_mean) / self.label_std
        y = self.model(features)
        loss = self.mse_loss(y, labels)
        if not self.isAutoML:
            self.log("train_mse_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        
        y = self.model(features)

        labels_normed = (labels - self.label_mean) / self.label_std
        loss_normed = self.L1_loss(y, labels_normed)
        if not self.isAutoML:
            self.log("val_L1_loss_normed", loss_normed)

        y = y * self.label_std + self.label_mean
        loss = self.L1_loss(y, labels)
        self.log("val_L1_loss", loss, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        features, labels, filename = batch
        features, labels = features.to(device), labels.to(device)
        
        y = self.model(features)
        y = y * self.label_std + self.label_mean
        loss = self.L1_loss(y, labels)
        self.log("test_L1_loss", loss, prog_bar=False)
        for i in range(y.size()[0]):
            filename_ = filename[i]
            y_ = y[i]
            self.test_step_outputs.append([filename_, y_])
        return loss

    def on_test_epoch_end(self):
        with open(self.csv_file,'w') as csv_writer:
            writer=csv.writer(csv_writer, delimiter='\t',lineterminator='\n',)
            for output in self.test_step_outputs:
                gazes = output[1].cpu().detach().numpy().flatten().tolist()
                name = output[0].split('/')[-1].split('.')[0].split('_')[-1]
                row = [name, gazes]
                writer.writerow(row)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description="train grace face net")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--feature_type", type=str, default='face_embed', choices=['face_embed', 'ldmk'])
        return parser.parse_args()

    path = '/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/dataset/updated_gau_1000_features/'
    log_dir = "/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/emotion/motor_prediction/lightning_logs/"
    log_name = "face_embed"
    args = parse_args()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, log_name))
    label_mean, label_std, ldmk_mean, ldmk_std = calculate_data_stat(path)
    print(f"data stat: label mean={label_mean}, label std={label_std}, ldmk mean={ldmk_mean}, ldmk_std={ldmk_std}")

    # training and testing
    model = GracePL(os.path.join(log_dir, log_name), label_mean, label_std,
                    args.learning_rate, args.feature_type)

    train_loader = DataLoader(GraceFaceDataset(image_path=path, split='train', feature_type=args.feature_type), 
                            batch_size=32, shuffle=True, num_workers=11, persistent_workers=True)
    test_loader = DataLoader(GraceFaceDataset(image_path=path, split='test', feature_type=args.feature_type), 
                            batch_size=50, shuffle=False)
    val_loader = DataLoader(GraceFaceDataset(image_path=path, split='val', feature_type=args.feature_type), 
                            batch_size=50, shuffle=False)

    early_stop_callback = EarlyStopping(monitor="val_L1_loss", patience=500, verbose=False, mode="min")
    trainer = L.Trainer(max_epochs=200000,
                        log_every_n_steps=10,
                        logger=tb_logger,
                        callbacks=[early_stop_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

    # only testing
    '''
    checkpoint = "grace_emo/emotion/motor_prediction/lightning_logs/5-layer NN/lightning_logs/version_0/checkpoints/epoch=6450-step=90314.ckpt"
    model = GracePL.load_from_checkpoint(checkpoint,
                                         output_path=os.path.join(log_dir, log_name),
                                         label_mean=label_mean, label_std=label_std, ldmk_mean=ldmk_mean, ldmk_std=ldmk_std,
                                         learning_rate=args.learning_rate
                                         )
    trainer = L.Trainer()
    test_loader = DataLoader(GraceFaceDataset(image_path=path, split='test'), batch_size=50, shuffle=False)
    model.eval()
    trainer.test(model, dataloaders=test_loader)'''
