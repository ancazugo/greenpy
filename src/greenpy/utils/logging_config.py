import logging
from pathlib import Path


def setup_logger(log_path: str | Path, log_level: str = logging.WARNING) -> None:
    logging.basicConfig(
        filename=log_path,
        encoding="utf-8",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        style="%",
        datefmt="%Y-%m-%d %H:%M",
        level=log_level,
    )
