from trainer import *
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
from IPython.utils import io

def parse_args():
    parser = argparse.ArgumentParser(description="train grace face net")
    parser.add_argument(
        "--log_path", type=str, required=True, help="dir to place tensorboard logs from all trials"
    )
    parser.add_argument("--node1", type=int, default=256)
    parser.add_argument("--node2", type=int, default=512)
    parser.add_argument("--node3", type=int, default=1024)
    parser.add_argument("--node4", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print('input arguments: ', args)
    path = '/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/dataset/processed_gau_600/'

    early_stop_callback = EarlyStopping(monitor="val_L1_loss", patience=500, verbose=False, mode="min")

    label_mean, label_std, ldmk_mean, ldmk_std = calculate_data_stat(path)
    model = GracePL(label_mean, label_std, ldmk_mean, ldmk_std,
                    args.node1, args.node2, args.node3, args.node4, args.learning_rate, autoML=True)

    train_loader = DataLoader(GraceFaceDataset(image_path=path, split='train'), batch_size=32, shuffle=True, num_workers=11, persistent_workers=True)
    test_loader = DataLoader(GraceFaceDataset(image_path=path, split='test'), batch_size=50, shuffle=False)
    val_loader = DataLoader(GraceFaceDataset(image_path=path, split='val'), batch_size=50, shuffle=False)

    trainer = L.Trainer(max_epochs=200000,
                        log_every_n_steps=10,
                        logger=False,
                        enable_progress_bar=False,
                        deterministic=True,
                        default_root_dir=args.log_path,
                        callbacks=[early_stop_callback])

    logger = pl_loggers.TensorBoardLogger(args.log_path)
    print(f"Logging to path: {args.log_path}.")

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,)
    #trainer.test(model, dataloaders=test_loader)

    with io.capture_output() as captured:
        val_loss = trainer.validate()[0]["val_L1_loss"]
    logger.log_metrics({"val_L1_loss": val_loss})

    # Log the number of model parameters
    num_params = trainer.model.num_params
    logger.log_metrics({"num_params": num_params})

    logger.save()

    # Print outputs
    print(f"Val loss: {val_loss}, num_params: {num_params}")