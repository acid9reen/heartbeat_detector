from typing import Any
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
    loss_name: Literal['mse', 'bce']


class OutputConfig(BaseModel):
    out_folder: str


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

    @property
    def dict_repr(self) -> dict[str, Any]:
        return {
            'model_name': self.model_name,
            'optimizer': self.optimizer_config.optimizer_name,
            'learning_rate': self.optimizer_config.learning_rate,
            'weight_decay': self.optimizer_config.weight_decay,
            'scheduler_milestones': self.scheduler_config.milestones,
            'scheduler_gamma': self.scheduler_config.gamma,
            'loss': self.loss_config.loss_name,
        }


def read_config(config_path: str) -> TrainConfig:
    with open(config_path, 'r') as in_:
        config = yaml.safe_load(in_)

    return TrainConfig(**config)
