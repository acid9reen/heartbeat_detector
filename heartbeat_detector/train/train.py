import inspect
from functools import partial
from pathlib import Path

import torch

from heartbeat_detector.configs.train_config import TrainConfig
from heartbeat_detector.dataset.dataset import HeartbeatDataloaders
from heartbeat_detector.models import UNet1d
from heartbeat_detector.train.model_saver import ModelSaver
from heartbeat_detector.train.trainer import Trainer


OPTIMIZERS = {
    'adam': torch.optim.Adam,
}

LOSSES = {
    'mse': torch.nn.MSELoss,
    'bce': partial(torch.nn.BCELoss, reduction='mean'),
}


def train(config: TrainConfig) -> None:
    train_dataloader, validation_dataloader = HeartbeatDataloaders(
        config.dataset_config.dataset_filepath,
        config.dataset_config.test_folds,
        config.dataset_config.batch_size,
        config.dataset_config.num_workers,
        config.dataset_config.validation_split_ratio,
        pin_memory=config.dataset_config.pin_memory,
    ).get_train_validation_dataloaders()

    model = UNet1d()
    optimizer = OPTIMIZERS[config.optimizer_config.optimizer_name](
        model.parameters(),
        config.optimizer_config.learning_rate,
        weight_decay=config.optimizer_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        config.scheduler_config.milestones,
        gamma=config.scheduler_config.gamma,
    )
    loss = LOSSES[config.loss_config.loss_name]()

    model = model.to(config.device)

    trainer = Trainer(
        model,
        train_dataloader,
        validation_dataloader,
        optimizer,
        scheduler,
        loss,
        config.device,
    )

    epochs_num = config.epochs_num

    model_saver = ModelSaver(
        config.output_config.out_folder,
        config.model_name,
        Path(inspect.getfile(UNet1d)),
    )

    for epoch_no, model in enumerate(trainer.train(epochs_num), start=1):
        if epoch_no % config.save_step_size == 0:
            model_saver.save(model, epoch_no)

        print(f'Epoch {epoch_no} / {epochs_num}')
