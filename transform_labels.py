import argparse
import json
import multiprocessing as mp
import os
import statistics
from functools import partial
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm


class Peak(NamedTuple):
    index: int
    channel: int


def read_label_file(filepath: Path) -> list[list[int]]:
    with open(filepath, 'r') as in_:
        labels = json.load(in_)

    return labels


def labels_to_pairs(labels: list[list[int]]) -> tuple[list[Peak], set[int]]:
    peaks = []
    empty_channels = set()

    for channel_index, channel in enumerate(labels):
        if channel:
            transformed_channel = map(lambda index: Peak(index, channel_index), channel)
            peaks.extend(transformed_channel)
            continue

        empty_channels.add(channel_index)

    return sorted(peaks, key=lambda peak: peak.index), empty_channels


def fix_peaks(
        peaks: list[Peak],
        empty_channels: set[int],
        window_size: int = 20,
) -> list[list[int]]:
    result: list[list[int]] = [[] for __ in range(64)]
    threshold = (64 - len(empty_channels)) // 2

    pivot = peaks[0].index
    window_frame: list[Peak] = []

    for peak in peaks:
        if (index := peak.index) < pivot + window_size:
            window_frame.append(peak)
            continue

        pivot = index

        if len(window_frame) > threshold:
            channel_index = {peak.channel: peak.index for peak in window_frame}
            mean = round(statistics.mean(channel_index.values()))

            for i in range(len(result)):
                if i in empty_channels:
                    continue

                result[i].append(channel_index.get(i, mean))

        window_frame: list[Peak] = []

    return result


def save_processed_labels(processed_labels: list[list[int]], filepath: Path) -> None:
    with open(filepath, 'w') as out:
        json.dump(processed_labels, out)


class LabelProcessor(object):
    def __init__(
            self,
            filepaths: list[Path],
            out_folder: str,
            window_size: int = 15,
            num_workers: int = mp.cpu_count() // 2,
    ) -> None:
        out_filepaths = [
            path.parent.parent / out_folder / path.name
            for path in filepaths
        ]

        if out_filepaths:
            os.makedirs(out_filepaths[0].parent, exist_ok=True)

        self.filepaths_out_filepaths = list(zip(filepaths, out_filepaths))
        self.peaks_fixer = partial(
            fix_peaks, window_size=window_size,
        )
        self.num_workers = num_workers

    def _process_single(self, filepath_out_filepath: tuple[Path, Path]) -> None:
        filepath, out_filepath = filepath_out_filepath

        labels = read_label_file(filepath)
        peaks, empty_channels = labels_to_pairs(labels)
        processed_labels = self.peaks_fixer(peaks, empty_channels)
        save_processed_labels(processed_labels, out_filepath)

    def process(self) -> None:
        with mp.Pool(self.num_workers) as pool:
            for __ in tqdm(
                    pool.imap_unordered(self._process_single, self.filepaths_out_filepaths),
                    desc='Processing labels',
                    total=len(self.filepaths_out_filepaths),
            ):
                ...


class LabelProcessorNamespace(argparse.Namespace):
    labels_data_root: str
    out_folder: str
    window_size: int = 20
    num_workers: int = mp.cpu_count() // 2


def parse_args() -> LabelProcessorNamespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('labels_data_root', help='Path to dataset labels')
    parser.add_argument('out_folder', help='Name of output folder')
    parser.add_argument('--num_workers', type=int, help='Number of parallel processes')
    parser.add_argument('--window_size', type=int, help='Window size for same peak detection')

    return parser.parse_args(namespace=LabelProcessorNamespace())


def main() -> int:
    args = parse_args()
    filepaths = [path for path in Path(args.labels_data_root).rglob('**/*.json')]

    labels_processor = LabelProcessor(
        filepaths, args.out_folder, args.window_size, args.num_workers,
    )

    labels_processor.process()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
