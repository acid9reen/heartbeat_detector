import logging

import mlflow
import numpy as np
import torch
from heartbeat_detector.types import Devices
from heartbeat_detector.losses.dice import BceDiceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger(__name__)


class TestModel(object):
    def __init__(
            self,
            model: torch.nn.Module,
            test_dataloader: DataLoader,
            device: Devices,
    ) -> None:
        self.model = model
        self.dataloader = test_dataloader
        self.device = device
        self.loss = BceDiceLoss()

    def _test_loop(self) -> float:
        running_loss = .0

        with torch.no_grad():
            for __, signals, labels in tqdm(
                    self.dataloader,
                    desc='Test batches',
                    total=len(self.dataloader),
            ):
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(signals)
                loss = self.loss(preds, labels)

                mean_loss: float = np.mean(loss.cpu().detach().numpy()).item()
                running_loss += mean_loss

        return running_loss / len(self.dataloader)

    def test(self) -> None:
        test_mean_loss = self._test_loop()

        logger.info(f'Test loss: {test_mean_loss:.4f}')
        mlflow.log_metric('test_loss', test_mean_loss, step=0)
