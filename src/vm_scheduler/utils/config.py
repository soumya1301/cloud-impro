"""
Utility modules for logging, seed setting, configuration loading, and path resolution.
"""
import os
import random
import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import yaml


def setup_logger(name: str = "vm_scheduler", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configures a standardized console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


def set_seed(seed: int = 42) -> None:
    """Enforces deterministic execution across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads and returns a YAML configuration file as a Python dictionary."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Returns the absolute root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent.parent

