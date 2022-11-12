import csv
import multiprocessing as mp
import sys
from functools import partial

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


Set = tuple[tuple, tuple]


def read_dataset_file(dataset_file_path: str) -> tuple[list[str], list[str], list[int]]:
    signal_files = []
    label_files = []
    num_peaks = []

    with open(dataset_file_path, 'r') as dataset:
        csv_reader = csv.reader(
            dataset,
            delimiter=',',
            quotechar='"',
        )

        # Skip header
        __, *data = csv_reader

        for row in data:
            signal_path, label_path, num_peaks_str = row

            signal_files.append(signal_path)
            label_files.append(label_path)
            num_peaks.append(int(num_peaks_str))

    return signal_files, label_files, num_peaks


class HeartbeatDataset(Dataset):
    def __init__(
            self,
            signal_files: tuple[str],
            label_files: tuple[str],
    ) -> None:
        super().__init__()
        self.signal_files = signal_files
        self.label_files = label_files

    def __len__(self) -> int:
        return len(self.signal_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = torch.from_numpy(np.load(self.signal_files[index]))
        label = torch.from_numpy(np.load(self.label_files[index]))

        return signal, label


class HeartbeatDataloaders(object):
    def __init__(
            self,
            dataset_file_path: str,
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

        signal_files, label_files, num_peaks = read_dataset_file(
            dataset_file_path,
        )

        (
            (self.signals_train, self.labels_train),
            (self.signals_validation, self.labels_validation),
            (self.signals_test, self.labels_test),
        ) = self._train_validation_test_split(
            signal_files, label_files, num_peaks,
        )

    @staticmethod
    def _train_validation_test_split(
            signal_files: list[str],
            label_files: list[str],
            num_peaks: list[int],
    ) -> tuple[Set, Set, Set]:
        # NOTE: Treating num_peaks as labels in train_test_split()
        # for stratification by number of peaks

        signals_labels = list(zip(signal_files, label_files))

        try:
            signals_labels_train, signals_labels_test, num_peaks_train, __ = train_test_split(
                signals_labels, num_peaks, test_size=0.2, stratify=num_peaks,
            )
        except ValueError as e:
            print(
                'Possible reason is poor quality data, '
                'remove rows with unique num_peaks values '
                'and try again',
                file=sys.stderr,
            )

            raise e

        signals_labels_train, signals_labels_validation, __, __ = train_test_split(
            signals_labels_train,
            num_peaks_train,
            test_size=0.25,
            stratify=num_peaks_train,
        )

        train_set = tuple(zip(*signals_labels_train))
        validation_set = tuple(zip(*signals_labels_validation))
        test_set = tuple(zip(*signals_labels_test))

        return train_set, validation_set, test_set

    def _get_train_dataloader(self) -> DataLoader:
        return self.pre_tuned_dataloader(
            HeartbeatDataset(
                self.signals_train, self.labels_train,
            ),
        )

    def _get_validation_dataloader(self) -> DataLoader:
        return self.pre_tuned_dataloader(
            HeartbeatDataset(
                self.signals_validation, self.labels_validation,
            ),
        )

    def _get_test_dataloader(self) -> DataLoader:
        return self.pre_tuned_dataloader(
            HeartbeatDataset(
                self.signals_test, self.labels_test,
            ),
        )

    def get_train_validation_test_dataloaders(
            self,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        return (
            self._get_train_dataloader(),
            self._get_validation_dataloader(),
            self._get_test_dataloader(),
        )
