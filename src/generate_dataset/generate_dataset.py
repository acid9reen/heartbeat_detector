Secs = int


class DatasetGenerator:
    def __init__(self, raw_data_path: str, trim_by: Secs, limit: int) -> None:
        self.raw_data_path = raw_data_path
        self.trim_by = trim_by
        self.limit = limit
