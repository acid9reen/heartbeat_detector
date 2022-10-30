import json
import logging
import random
from functools import partial
from pathlib import Path

from .selector import SelectInstruction


logger = logging.getLogger(__name__)


Secs = int


def _get_labels_per_channel_peaks_length(
        labels_paths: list[Path],
) -> dict[Path, list[int]]:
    label_path_peaks_length: dict[Path, list[int]] = {}

    for file_path in labels_paths:
        with open(str(file_path), 'r') as label_file:
            peaks = json.load(label_file)
            peaks_lengths = list(map(len, peaks))
            label_path_peaks_length[file_path] = peaks_lengths

    return label_path_peaks_length


def _get_signal_path_from_label_path(label_path: Path, signals_location: str) -> Path:
    # Initial implementation for the following dataset structure:
    # dataset_root
    # │
    # └-X
    # │ └-X_<...>.npy
    # │ └-X_<...>.npy
    # │ ...
    # │
    # └-Y
    #   └-Y_<...>.json
    #   └-Y_<...>.json
    #   ...

    label_filename = label_path.stem
    signal_filaname = ''.join(['X', label_filename[1:], '.npy'])

    return label_path.parent.parent / signals_location / signal_filaname


class DatasetGenerator:
    SIGNALS_LOCATION = 'X'
    LABELS_LOCATION = 'Y'
    FREQUENCY = 5000
    N_CHANNELS = 64

    def __init__(
            self,
            raw_data_path: str,
            trim_by: Secs,
            limit: int,
            out_folder_path: str,
    ) -> None:
        self.raw_data_path = Path(raw_data_path)
        self.sample_length = trim_by * self.FREQUENCY
        self.limit = limit
        self.out_foder_path = out_folder_path
        self.label_paths = self._get_labels_files()
        self.labels_per_channel_peaks_length = _get_labels_per_channel_peaks_length(
            self.label_paths,
        )
        self.get_signal_path_from_label_path = partial(
            _get_signal_path_from_label_path,
            signals_location=self.SIGNALS_LOCATION,
        )

    def _get_labels_files(self) -> list[Path]:
        labels_path = self.raw_data_path / self.LABELS_LOCATION
        label_files = [filepath for filepath in labels_path.glob('*.json')]

        logger.info(f'Found {len(label_files)} label files in {labels_path}')

        return label_files

    def _generate_select_insruction(self) -> SelectInstruction:
        label_file_path = random.choice(self.label_paths)

        peaks_lengths = self.labels_per_channel_peaks_length[label_file_path]
        non_empty_channels = [
            index for index, length in enumerate(peaks_lengths) if length > 0
        ]
        channel_index = random.choice(non_empty_channels)
        peak_index = random.randint(0, peaks_lengths[channel_index] - 1)
        shift = random.randint(0, self.sample_length - 1)

        return SelectInstruction(
            label_file_path,
            channel_index,
            peak_index,
            shift,
        )

    def _get_select_instructions(self) -> list[SelectInstruction]:
        select_instructions: list[SelectInstruction] = []
        for __ in range(self.limit):
            select_instruction = self._generate_select_insruction()
            select_instructions.append(select_instruction)

        return select_instructions
