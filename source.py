import argparse
import logging
import os
import random
from typing import Callable

import mlflow
import numpy as np
import torch

from src.configs.train_config import read_config
from src.generate_dataset import DatasetGenerator
from src.generate_dataset.label_transformers import IdentityTransformer
from src.generate_dataset.label_transformers import TriangleTransformer
from src.generate_dataset.label_transformers import WaveTransformer
from src.train.train import train


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

TRANSFORM_KEYS = {
    'identity': IdentityTransformer(),
    'sin': WaveTransformer(20),
    'abs': TriangleTransformer(20),
}


class ParserNamespace(argparse.Namespace):
    # <<---- General options ---->>
    random_seed: int = 420

    # <<---- Dataset generator options ---->>
    trim_by: Secs
    limit: int
    raw_data_root: str
    output_folder: str = './dataset'

    # Store function in context of used subparser
    # NOTE: You need to set default for this field for any subparser you add!
    action: Callable[['ParserNamespace'], None]


def parse_args() -> ParserNamespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # <<---- General arguments ---->>
    parser.add_argument(
        '--random_seed', help='Fixed random seed',
    )

    # <<---- Generate dataset arguments ---->>
    generate_dataset_parser = subparsers.add_parser(
        'generate_dataset',
        help='Script to generate dataset from .npy and .json files',
    )

    generate_dataset_parser.add_argument(
        '--trim_by',
        type=int,
        help='Desired length of a signal to generate in seconds',
    )

    generate_dataset_parser.add_argument(
        '--limit',
        type=int,
        help='Number of signals to generate (size of desired dataset)',
    )

    generate_dataset_parser.add_argument(
        '--label_transform',
        type=str,
        choices=['identity', 'sin', 'abs'],
        help='Type of label transform',
    )

    generate_dataset_parser.add_argument(
        '--raw_data_root', help='Path to files to generate dataset from',
    )

    generate_dataset_parser.add_argument(
        '--output_folder', help='Location to store generated dataset',
    )

    train_model_parser = subparsers.add_parser('train_model', help='Script to train model')
    train_model_parser.add_argument('train_config_path', help='Path to train config .yaml file')

    train_model_parser.set_defaults(action=train_model)

    generate_dataset_parser.set_defaults(action=generate_dataset)

    return parser.parse_args(namespace=ParserNamespace())


def generate_dataset(args: ParserNamespace) -> None:
    """Create dataset generator object for now

    Args:
        args (ParserNamespace): Args for dataset generation
    """

    seed_everything(args.random_seed)

    dataset_generator = DatasetGenerator(
        args.raw_data_root,
        args.trim_by,
        args.limit,
        args.output_folder,
    )

    dataset_generator.generate(TRANSFORM_KEYS[args.label_transform])


def train_model(args: ParserNamespace) -> None:
    config = read_config(args.train_config_path)

    with mlflow.start_run():
        seed_everything(args.random_seed)
        mlflow.log_params(config.dict_repr)
        train(config)


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


def main(args: ParserNamespace) -> int:
    args.action(args)

    return 0


if __name__ == '__main__':
    args = parse_args()
    main(args)
