from typing import Protocol

import numpy as np


class LabelTransformer(Protocol):
    def transform(self, label: np.ndarray) -> np.ndarray:
        ...


class IdentityTransformer(object):
    """Fake transformer, just for default value
    """

    def transform(self, label: np.ndarray) -> np.ndarray:
        return label


class WaveTransformer(object):
    """Transform signal to form of wave (like sin or cos) around ones in original one
         ^                       -
         |       ---->       --/   \--
    -----|-----         ----/         \----
    """

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def transform(self, label: np.ndarray) -> np.ndarray:
        peaks_indexes = np.where(label == 1)[0]
        half_window_size = 20

        sin = []

        for i in range(2 * half_window_size + 1):
            x = i / (2 * half_window_size + 1)
            sin.append((np.sin(2 * np.pi * x - np.pi / 2) + 1) / 2)

        sin = np.array(sin)

        for peak_index in peaks_indexes:

            if peak_index - half_window_size < 0:
                right_slice_boarder = peak_index + half_window_size

                label[:right_slice_boarder] = sin[len(sin) - (right_slice_boarder):]

            elif peak_index + half_window_size > len(label):
                left_slice_boarder = peak_index - half_window_size - 1

                label[left_slice_boarder:] = sin[:len(label[left_slice_boarder:])]
            else:
                left_slice_boarder = peak_index - half_window_size - 1
                right_slice_boarder = peak_index + half_window_size

                label[left_slice_boarder:right_slice_boarder] = sin

        return label


class TriangleTransformer(object):
    """Transform signal to form of triangle around ones in original one
         ^                    ^
         |       ---->       / \
    -----|-----         ----/   \----
    """

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def transform(self, label: np.ndarray) -> np.ndarray:
        peaks_indexes = np.where(label == 1)[0]
        half_window_size = 20

        absolute = []

        for i in range(2 * half_window_size + 1):
            x = i / (2 * half_window_size + 1)
            absolute.append(-2 * np.abs(x - 0.5) + 1)

        absolute = np.array(absolute)

        for peak_index in peaks_indexes:

            if peak_index - half_window_size < 0:
                right_slice_boarder = peak_index + half_window_size

                label[:right_slice_boarder] = absolute[len(absolute) - (right_slice_boarder):]

            elif peak_index + half_window_size > len(label):
                left_slice_boarder = peak_index - half_window_size - 1

                label[left_slice_boarder:] = absolute[:len(label[left_slice_boarder:])]
            else:
                left_slice_boarder = peak_index - half_window_size - 1
                right_slice_boarder = peak_index + half_window_size

                label[left_slice_boarder:right_slice_boarder] = absolute

        return label
