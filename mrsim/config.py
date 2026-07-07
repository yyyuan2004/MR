"""YAML config loading, run-directory management, and seeding."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"config {path} must contain a mapping at the top level")
    return config


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch; return a NumPy generator for local use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def run_dir(config: dict[str, Any]) -> Path:
    """Return (and create) runs/<experiment_name>/ for this config."""
    out = Path("runs") / str(config["experiment_name"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_config_snapshot(config: dict[str, Any], out_dir: Path, name: str) -> Path:
    """Write a JSON snapshot of the config used by one script."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"config_{name}.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path
