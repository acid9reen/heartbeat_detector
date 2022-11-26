from functools import partial

import torch

from .model_saver import ModelSaver
from .trainer import Trainer
from src.configs.train_config import TrainConfig
from src.dataset.dataset import HeartbeatDataloaders
from src.models import UNet1d


OPTIMIZERS = {
    'adam': torch.optim.Adam,
}

LOSSES = {
    'mse': torch.nn.MSELoss,
    'bce': partial(torch.nn.BCELoss, reduction='mean'),
}


def train(config: TrainConfig) -> None:
    train_dataloader, validation_dataloader, __ = HeartbeatDataloaders(
        config.dataset_config.dataset_filepath,
        config.dataset_config.batch_size,
        config.dataset_config.num_workers,
        pin_memory=config.dataset_config.pin_memory,
    ).get_train_validation_test_dataloaders()

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
    )

    for epoch_no, model in enumerate(trainer.train(epochs_num), start=1):
        if epoch_no % config.save_step_size == 0:
            model_saver.save(model, epoch_no)

        print(f'Epoch {epoch_no} / {epochs_num}')
