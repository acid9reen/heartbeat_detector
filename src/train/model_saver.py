import os

import torch
from torch import nn


class ModelSaver(object):
    def __init__(self, out_folder: str, model_name: str) -> None:
        self.save_path_template = os.path.join(out_folder, model_name)

        os.makedirs(out_folder, exist_ok=True)

    def save(self, model: nn.Module, epoch_no: int) -> None:
        epoch_specific_addon = f'_epoch_{epoch_no:03d}.pth'
        torch.save(model, self.save_path_template + epoch_specific_addon)
