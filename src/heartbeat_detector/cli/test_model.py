import argparse
import logging
from pathlib import Path
from typing import get_args as get_typing_args

import mlflow
import torch
from heartbeat_detector.dataset.dataset import HeartbeatDataloaders
from heartbeat_detector.test import TestModel
from heartbeat_detector.types import Devices

LOG_FILE = 'debug.log'


# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
    ],
)


logger = logging.getLogger(__name__)


class TestModelNamespace(argparse.Namespace):
    model_path: Path
    run_id: str
    dataset_path: Path
    batch_size: int
    test_folds: str
    device: Devices


def parse_test_model_args() -> TestModelNamespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_path', type=Path, help='Path to model in .pth format',
    )
    parser.add_argument(
        'run_id', help='MLFlow run identifier',
    )
    parser.add_argument(
        'dataset_path', type=Path, help='Path to dataset .csv file',
    )
    parser.add_argument(
        'test_folds', help='Comma separated test folds (\'aaa,bbb,ccc\')',
    )
    parser.add_argument(
        '-b', '--batch_size', type=int, help='Batch size for inference', default=600,
    )
    parser.add_argument(
        '-d',
        '--device',
        choices=get_typing_args(Devices),
        help='Device for inference',
        default='cuda',
    )

    return parser.parse_args(namespace=TestModelNamespace())


def main() -> int:
    args = parse_test_model_args()

    model = torch.load(args.model_path.as_posix(), map_location=args.device)
    model = model.eval()

    test_dataloader = HeartbeatDataloaders(
        args.dataset_path.as_posix(),
        args.test_folds.split(','),
        args.batch_size,
        num_workers=1,
    ).get_test_dataloader()

    model_tester = TestModel(model, test_dataloader, device=args.device)

    with mlflow.start_run(run_id=args.run_id):
        model_tester.test()

    return 0
