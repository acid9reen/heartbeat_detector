import argparse
import logging
from typing import Callable

from src.generate_dataset import DatasetGenerator


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

    # Generate dataset arguments
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
        '--raw_data_root', help='Path to files to generate dataset from',
    )

    generate_dataset_parser.add_argument(
        '--output_folder', help='Location to store generated dataset',
    )

    generate_dataset_parser.set_defaults(action=generate_dataset)

    return parser.parse_args(namespace=ParserNamespace())


def generate_dataset(args: ParserNamespace) -> None:
    """Create dataset generator object for now

    Args:
        args (ParserNamespace): Args for dataset generation
    """

    DatasetGenerator(
        args.raw_data_root,
        args.trim_by,
        args.limit,
        args.output_folder,
    )


def main(args: ParserNamespace) -> int:
    args.action(args)

    return 0


if __name__ == '__main__':
    args = parse_args()
    main(args)
