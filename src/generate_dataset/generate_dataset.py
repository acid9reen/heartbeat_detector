import logging
from pathlib import Path


logger = logging.getLogger(__name__)


Secs = int


class DatasetGenerator:
    SIGNALS_LOCATION = 'X'
    LABELS_LOCATION = 'Y'
    FREQUENCY = 5000
    N_CHANNELS = 64

    def __init__(
            self,
            raw_data_path: str,
            trim_by: Secs,
            limit: int,
            out_folder_path: str,
    ) -> None:
        self.raw_data_path = Path(raw_data_path)
        self.sample_length = trim_by * self.FREQUENCY
        self.limit = limit
        self.out_foder_path = out_folder_path

    def _get_signal_path_from_label_path(self, label_path: Path) -> Path:
        # Initial implementation for the following dataset structure:
        # dataset_root
        # │
        # └-X
        # │ └-X_<...>.npy
        # │ └-X_<...>.npy
        # │ ...
        # │
        # └-Y
        #   └-Y_<...>.json
        #   └-Y_<...>.json
        #   ...

        label_filename = label_path.stem
        signal_filaname = ''.join(['X', label_filename[1:], '.npy'])

        return label_path.parent.parent / self.SIGNALS_LOCATION / signal_filaname

    def _get_labels_files(self) -> list[Path]:
        labels_path = self.raw_data_path / self.LABELS_LOCATION
        label_files = [filepath for filepath in labels_path.glob('*.json')]

        logger.info(f'Found {len(label_files)} label files in {labels_path}')

        return label_files
