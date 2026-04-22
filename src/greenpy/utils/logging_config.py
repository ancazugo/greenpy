import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_path: str | Path, log_level: str = "WARNING") -> None:
    fmt = "{time:YYYY-MM-DD HH:mm} - {level} - {message}"
    logger.remove()
    logger.add(sys.stderr, level=log_level, format=fmt)
    logger.add(log_path, level=log_level, encoding="utf-8", format=fmt)
