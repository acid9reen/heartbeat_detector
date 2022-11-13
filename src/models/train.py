import logging
from typing import Any
from typing import Generator
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(
            self,
            model: nn.Module,
            train_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            optimizer: Any,
            scheduler: Any,
            loss: Any,
            device: Literal['cuda', 'cpu'],
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.device = device

        # TODO: move to descriptors
        self._train_batch_no = 0
        self._validation_batch_no = 0

    @property
    def train_batch_no(self) -> int:
        current = self._train_batch_no
        self._train_batch_no += 1

        return current

    @property
    def validation_batch_no(self) -> int:
        current = self._validation_batch_no
        self._validation_batch_no += 1

        return current

    def _train_loop(self) -> float:
        running_loss = 0.0
        self.model.train()

        for signals, labels in tqdm(
                self.train_dataloader,
                desc='Train batches',
                total=len(self.train_dataloader),
        ):
            self.optimizer.zero_grad()

            signals = signals.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(signals)
            loss = self.loss(preds, labels)
            loss.backward()

            self.optimizer.step()

            mean_loss = np.mean(loss.cpu().detach().numpy())
            running_loss += mean_loss.item()

            # TODO: add batch loss logging

        self.scheduler.step()

        return running_loss / len(self.train_dataloader)

    def _validation_loop(self) -> float:
        running_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for signals, labels in tqdm(
                    self.validation_dataloader,
                    desc='Train batches',
                    total=len(self.validation_dataloader),
            ):
                self.optimizer.zero_grad()

                signals = signals.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(signals)
                loss = self.loss(preds, labels)

                mean_loss = np.mean(loss.cpu().detach().numpy())
                running_loss += mean_loss.item()

                # TODO: add batch loss logging

        return running_loss / len(self.validation_dataloader)

    def train(self, epoch_num: int) -> Generator[nn.Module, None, None]:
        for epoch_no in range(epoch_num):
            train_mean_loss = self._train_loop()
            validation_mean_loss = self._validation_loop()

            logger.info(f'Epoch: {epoch_no}, train loss: {train_mean_loss:.4f}')
            logger.info(f'Epoch: {epoch_no}, validation loss: {validation_mean_loss:.4f}')

            yield self.model
