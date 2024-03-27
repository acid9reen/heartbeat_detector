import csv
import logging
import multiprocessing as mp
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Iterable, TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class Stage3Schema(TypedDict):
    signal: str
    label: str
    experiment: str
    channel: int


STAGE3_FIELDNAMES = sorted(list(Stage3Schema.__required_keys__))


def read_dataset_file(dataset_file_path: str) -> dict[str, list[Stage3Schema]]:
    """Read dataset file, store all dataset file rows in dict of lists,
    for now dict keys are:
        `x_file_path`,
        `y_file_path`,
        `num_peaks`,
        `channel`,

    Parameters
    ----------
    dataset_file_path : str
        Path to .csv dataset file

    Returns
    -------
    dataset : dict[str, list[dict[str, str]]]
        Dataset dict with keys storing casefolded original label file stem and
        values storing rows of the dataset
    """

    dataset: dict[str, list[Stage3Schema]] = defaultdict(list)

    with open(dataset_file_path, 'r', newline="") as dataset_file:
        csv_reader = csv.DictReader(
            dataset_file,
            fieldnames=STAGE3_FIELDNAMES,
        )

        # Skip header
        _ = next(csv_reader)

        row: Stage3Schema
        for row in csv_reader:  # type: ignore
            label_filename = row["experiment"]
            dataset[label_filename].append(row)

    logger.info(f'Find {len(dataset.keys())} folds in dataset, they are {", ".join(dataset.keys())}')

    return dataset


class HeartbeatDataset(Dataset):
    def __init__(
            self,
            signal_files: list[str],
            label_files: list[str],
    ) -> None:
        super().__init__()
        self.signal_files = signal_files
        self.label_files = label_files

    def __len__(self) -> int:
        return len(self.signal_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = torch.from_numpy(np.array([np.load(self.signal_files[index])], dtype=np.float32))
        label = torch.from_numpy(
            np.clip(
                np.array([np.load(self.label_files[index])], dtype=np.float32), 0, 1
            )
        )

        return signal, label


class HeartBeatDatasetWFilenames(HeartbeatDataset):
    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        signal, label = super().__getitem__(index)
        filename = self.signal_files[index]

        return filename, signal, label


class HeartbeatDataloaders(object):
    def __init__(
            self,
            dataset_file_path: str,
            test_folds: Iterable[str],
            validation_folds: set[str] = set(),
            exclude_folds: set[str] = set(),
            batch_size: int = 120,
            num_workers: int = mp.cpu_count() // 2,
            *,
            pin_memory: bool = True,
    ) -> None:
        self.pre_tuned_dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.dataset_root = Path(dataset_file_path).parent

        self.dataset = read_dataset_file(dataset_file_path)
        self.test_folds = set(map(str.casefold, test_folds))
        self.validation_folds = set(map(str.casefold, validation_folds))
        self.exclude_folds = set(map(str.casefold, exclude_folds))
        self.train_folds = (
            set(map(str.casefold, self.dataset.keys()))
            - self.test_folds
            - self.validation_folds
            - self.exclude_folds
        )

        logger.info('Done splitting data')
        logger.info(f'Train folds: {", ".join(self.train_folds)}')
        logger.info(f'Validation folds: {", ".join(self.validation_folds)}')
        logger.info(f'Test folds: {", ".join(self.test_folds)}')

    def _get_signals_labels_from_dataset(
            self,
            folds: Iterable[str],
    ) -> tuple[list[str], list[str]]:
        filtered_rows: list[Stage3Schema] = []

        for fold in folds:
            filtered_rows.extend(self.dataset[fold])

        signals = [(self.dataset_root / row['signal']).as_posix() for row in filtered_rows]
        labels = [(self.dataset_root / row['label']).as_posix() for row in filtered_rows]

        return signals, labels

    def _get_dataloader(
            self,
            signals: list[str],
            labels: list[str],
            dataset: type[HeartbeatDataset] = HeartbeatDataset,
    ) -> DataLoader:
        return self.pre_tuned_dataloader(
            dataset(signals, labels),
        )

    def get_train_validation_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        signals_train, labels_train = self._get_signals_labels_from_dataset(self.train_folds)
        signals_validation, labels_validation = self._get_signals_labels_from_dataset(self.validation_folds)

        logger.info(
            f'There are {len(signals_train)} train samples '
            f'and {len(signals_validation)} validation samples',
        )

        return (
            self._get_dataloader(signals_train, labels_train),
            self._get_dataloader(signals_validation, labels_validation),
        )

    def get_test_dataloader(self) -> DataLoader:
        signals, labels = self._get_signals_labels_from_dataset(self.test_folds)

        return self._get_dataloader(signals, labels, HeartBeatDatasetWFilenames)
