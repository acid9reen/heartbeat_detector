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
        ...


class TriangleTransformer(object):
    """Transform signal to form of triangle around ones in original one
         ^                    ^
         |       ---->       / \
    -----|-----         ----/   \----
    """

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def transform(self, label: np.ndarray) -> np.ndarray:
        ...
