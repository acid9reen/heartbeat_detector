import argparse
import logging
import os
import random

import mlflow
import numpy as np
import torch

from heartbeat_detector.configs.train_config import read_config
from heartbeat_detector.train.train import train


LOG_FILE = 'debug.log'


# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
    ],
)

# Now you can use same config for other files triggered by this module
# Just paste the line below into desired module
logger = logging.getLogger(__name__)


Secs = int


class ParserNamespace(argparse.Namespace):
    random_seed: int = 420
    train_config_path: str


def parse_args() -> ParserNamespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--random_seed', help='Fixed random seed',
    )

    parser.add_argument('train_config_path', help='Path to train config .yaml file')

    return parser.parse_args(namespace=ParserNamespace())


def seed_everything(seed: int) -> None:
    """Fix seed for random generators

    Args:
        seed (int): fixed seed
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    logger.info(f'Fix random seed with value: {seed}!')


def main() -> int:
    args = parse_args()

    config = read_config(args.train_config_path)

    with mlflow.start_run():
        seed_everything(args.random_seed)
        mlflow.log_params(config.dict_repr)
        train(config)

    return 0


if __name__ == '__main__':
    main()
