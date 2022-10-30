from pathlib import Path
from typing import Any
from typing import Callable
from typing import NamedTuple

import numpy as np


class SelectInstruction(NamedTuple):
    label_path: Path
    channel_index: int
    peak_index: int
    shift: int


class SelectedResult(NamedTuple):
    selected_signal: np.ndarray
    selected_labels: np.ndarray
    peak_quantity: int
    sample_file_name: str


class Selector:
    def __init__(
            self,
            label_transform: Callable[[list[int]], np.ndarray],
            sample_length: int,
    ) -> None:
        self.label_transform = label_transform
        self.sample_length = sample_length

    def select(
            self,
            signal: np.ndarray,
            label: list[int],
            select_instruction: SelectInstruction,
    ) -> Any:

        peak = label[select_instruction.peak_index]

        max_index = len(signal) - 1

        left_index = max(0, peak - select_instruction.shift)
        right_index = min(
            max_index,
            peak - select_instruction.shift + self.sample_length,
        )

        if (length := right_index - left_index) < self.sample_length:
            raise ValueError(
                f'Too short signal length: {length}, '
                f'while desired length {self.sample_length}',
            )

        selected_signal = signal[left_index:right_index]

        selected_peaks = list(filter(lambda x: left_index < x < right_index, label))

        selected_labels = np.zeros(len(selected_signal))

        for selected_peak in selected_peaks:
            selected_labels[selected_peak] = 1

        sample_file_name = (
            f'{select_instruction.label_path.stem}_{select_instruction.channel_index}'
            f'_{select_instruction.peak_index}_{select_instruction.shift}'
        )

        return SelectedResult(
            selected_signal,
            selected_labels,
            len(selected_peaks),
            sample_file_name,
        )
