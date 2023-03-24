import argparse
import logging

import mlflow

from heartbeat_detector.configs.train_config import read_config
from heartbeat_detector.train.train import train
from heartbeat_detector.utils import seed_everything


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


class TrainModelNamespace(argparse.Namespace):
    random_seed: int = 420
    train_config_path: str


def parse_train_model_args() -> TrainModelNamespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--random_seed', help='Fixed random seed',
    )

    parser.add_argument('train_config_path', help='Path to train config .yaml file')

    return parser.parse_args(namespace=TrainModelNamespace())


def main() -> int:
    args = parse_train_model_args()

    config = read_config(args.train_config_path)

    with mlflow.start_run():
        seed_everything(args.random_seed)
        mlflow.log_params(config.dict_repr)
        train(config)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
