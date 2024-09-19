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

device = get_device()

# define the LightningModule
class GracePL(L.LightningModule):
    def __init__(self, label_mean, label_std, ldmk_mean, ldmk_std,
                 node1, node2, node3, node4, learning_rate, autoML=False):
        super().__init__()
        self.model = GraceModel(node1, node2, node3, node4)
        self.mse_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        self.label_mean = label_mean
        self.label_std = label_std
        self.ldmk_mean = ldmk_mean
        self.ldmk_std = ldmk_std
        self.lr = learning_rate
        self.isAutoML = autoML

    def training_step(self, batch, batch_idx):
        images, ldmks, labels = batch
        images, ldmks, labels = images.to(device), ldmks.to(device), labels.to(device)
        ldmks = (ldmks - self.ldmk_mean) / self.ldmk_std
        labels = (labels - self.label_mean) / self.label_std
        y = self.model(ldmks)
        loss = self.mse_loss(y, labels)
        if not self.isAutoML:
            self.log("train_mse_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, ldmks, labels = batch
        images, ldmks, labels = images.to(device), ldmks.to(device), labels.to(device)
        ldmks = (ldmks - self.ldmk_mean) / self.ldmk_std
        y = self.model(ldmks)

        labels_normed = (labels - self.label_mean) / self.label_std
        loss_normed = self.L1_loss(y, labels_normed)
        if not self.isAutoML:
            self.log("val_L1_loss_normed", loss_normed)

        y = y * self.label_std + self.label_mean
        loss = self.L1_loss(y, labels)
        self.log("val_L1_loss", loss, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        images, ldmks, labels = batch
        images, ldmks, labels = images.to(device), ldmks.to(device), labels.to(device)
        ldmks = (ldmks - self.ldmk_mean) / self.ldmk_std
        y = self.model(ldmks)
        y = y * self.label_std + self.label_mean
        loss = self.L1_loss(y, labels)
        self.log("test_L1_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description="train grace face net")
        parser.add_argument("--node1", type=int, default=256)
        parser.add_argument("--node2", type=int, default=512)
        parser.add_argument("--node3", type=int, default=1024)
        parser.add_argument("--node4", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser.parse_args()

    path = 'grace_emo/dataset/processed_gau_600/'
    log_dir = "grace_emo/emotion/motor_prediction/lightning_logs/"
    log_name = "5-layer NN_dropout"

    args = parse_args()
    #checkpoint = "grace_emo/emotion/motor_prediction/lightning_logs/3-layer NN/lightning_logs/version_2/checkpoints/epoch=4999-step=70000.ckpt"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, log_name))
    label_mean, label_std, ldmk_mean, ldmk_std = calculate_data_stat(path)
    model = GracePL(label_mean, label_std, ldmk_mean, ldmk_std,
                    args.node1, args.node2, args.node3, args.node4, args.learning_rate)

    train_loader = DataLoader(GraceFaceDataset(image_path=path, split='train'), batch_size=32, shuffle=True, num_workers=11, persistent_workers=True)
    test_loader = DataLoader(GraceFaceDataset(image_path=path, split='test'), batch_size=50, shuffle=False)
    val_loader = DataLoader(GraceFaceDataset(image_path=path, split='val'), batch_size=50, shuffle=False)

    early_stop_callback = EarlyStopping(monitor="val_L1_loss", patience=500, verbose=False, mode="min")
    trainer = L.Trainer(max_epochs=200000, # still too small?
                        log_every_n_steps=10,
                        logger=tb_logger,
                        callbacks=[early_stop_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,)
                #ckpt_path=checkpoint)
    trainer.test(model, dataloaders=test_loader)