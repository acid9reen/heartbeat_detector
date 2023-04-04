import logging
import os
import random

import numpy as np
import torch


logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Fix seed for random generators

    Parameters
    ----------
    seed : int
        Seed to fix
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    logger.info(f'Fix random seed with value: {seed}!')
