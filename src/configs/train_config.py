from typing import Literal

import yaml
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    dataset_filepath: str
    num_workers: int
    batch_size: int
    pin_memory: bool


class OptimizerConfig(BaseModel):
    optimizer_name: Literal['adam']
    learning_rate: float
    weight_decay: float


class SchedulerConfig(BaseModel):
    milestones: list[int]
    gamma: float


class LossConfig(BaseModel):
    loss_name: Literal['mse']


class OutputConfig(BaseModel):
    checkpoints_folder: str


class TrainConfig(BaseModel):
    model_name: str
    dataset_config: DatasetConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    loss_config: LossConfig
    epochs_num: int
    save_step_size: int
    output_config: OutputConfig
    device: Literal['cpu', 'cuda']


def read_config(config_path: str) -> TrainConfig:
    with open(config_path, 'r') as in_:
        config = yaml.safe_load(in_)

    return TrainConfig(**config)
