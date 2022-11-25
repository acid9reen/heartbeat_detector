import os

import mlflow
import torch
from torch import nn


class ModelSaver(object):
    def __init__(self, out_folder: str, model_name: str) -> None:
        run_name = ''

        if (active_run := mlflow.active_run()) is not None:
            run_name = active_run.info.run_name

        out_folder_with_run_name = os.path.join(out_folder, run_name, 'checkpoints')

        self.save_path_template = os.path.join(out_folder_with_run_name, model_name)

        os.makedirs(out_folder_with_run_name, exist_ok=True)

    def save(self, model: nn.Module, epoch_no: int) -> None:
        epoch_specific_addon = f'_epoch_{epoch_no:03d}.pth'
        torch.save(model, self.save_path_template + epoch_specific_addon)
