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


class Selector:
    def __init__(self, label_transform: Callable[[list[int]], np.ndarray]) -> None:
        self.label_transform = label_transform

    def select(
            self,
            signal: np.ndarray,
            label: list[int],
            select_instruction: SelectInstruction,
    ) -> Any:
        ...
