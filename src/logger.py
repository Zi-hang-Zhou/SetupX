"""Unified logging: full DEBUG to file, ERROR-only to stderr."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import get_config


class LoggerSetup:
    _initialized = False

    @classmethod
    def setup(cls) -> logging.Logger:
        if cls._initialized:
            return logging.getLogger("setup_agent")

        config = get_config()
        env_log_file = os.getenv("LOG_FILE")
        env_log_dir = os.getenv("LOG_DIR")
        env_prefix = os.getenv("LOG_FILE_PREFIX", "").strip()

        log_dir = Path(env_log_dir) if env_log_dir else config.log_dir

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if env_log_file:
            log_file = Path(env_log_file)
        else:
            prefix = f"{env_prefix}_" if env_prefix else ""
            log_file = log_dir / f"{prefix}{timestamp}.log"

        logger = logging.getLogger("setup_agent")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console_handler)

        cls._initialized = True
        logger.info(f"logger initialized; log file: {log_file}")
        return logger


def get_logger(name: str | None = None) -> logging.Logger:
    LoggerSetup.setup()
    if name:
        return logging.getLogger(f"setup_agent.{name}")
    return logging.getLogger("setup_agent")
