"""YAML config loading, run-directory management, and seeding."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"config {path} must contain a mapping at the top level")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate the invariants shared by every experiment entry point."""
    for key in ("experiment_name", "seed", "data", "measurement", "mask", "recon"):
        if key not in config:
            raise ValueError(f"missing required config key: {key}")

    data = config["data"]
    n_images = int(data["n_images"])
    n_train = int(data["n_train"])
    n_val = int(data.get("n_val", 0))
    n_test = int(data["n_test"])
    image_size = int(data["image_size"])
    if n_images < 1 or image_size < 4:
        raise ValueError("data.n_images must be positive and data.image_size must be at least 4")
    if min(n_train, n_val, n_test) < 0 or n_train < 1 or n_test < 1:
        raise ValueError("data split sizes must be non-negative, with non-empty train and test splits")
    if n_train + n_val + n_test > n_images:
        raise ValueError("n_train + n_val + n_test exceeds data.n_images")

    fraction = float(config["mask"]["sampling_fraction"])
    if not 0.0 < fraction <= 1.0:
        raise ValueError("mask.sampling_fraction must be in (0, 1]")
    center_fraction = float(config["mask"].get("center_fraction", 0.0))
    if not 0.0 <= center_fraction <= 1.0:
        raise ValueError("mask.center_fraction must be in [0, 1]")
    noise_std = float(config["measurement"].get("noise_std", 0.0))
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("measurement.noise_std must be finite and non-negative")
    declared_mask_list = config["mask"].get("types", [])
    if not isinstance(declared_mask_list, list) or not declared_mask_list:
        raise ValueError("mask.types must be a non-empty list")
    if not all(isinstance(name, str) and name for name in declared_mask_list):
        raise ValueError("mask.types entries must be non-empty strings")
    declared_masks = set(declared_mask_list)
    line_masks = {
        "equispaced_lines",
        "variable_density_lines",
        "line_aopt",
        "line_subspace_leakage",
        "spectrum_energy_greedy",
        "recon_in_loop_greedy",
    }
    point_budget = max(1, int(round(fraction * image_size * image_size)))
    if declared_masks & line_masks and point_budget < image_size:
        raise ValueError(
            "the declared Cartesian line masks require a budget of at least "
            "one complete image column"
        )

    sweep = config.get("budget_sweep")
    if sweep is not None:
        accelerations = sweep.get("accelerations", [])
        if not isinstance(accelerations, list) or not accelerations:
            raise ValueError(
                "budget_sweep.accelerations must be a non-empty list"
            )
        acceleration_values = np.asarray(accelerations, dtype=np.float64)
        if (
            acceleration_values.ndim != 1
            or not np.isfinite(acceleration_values).all()
            or np.any(acceleration_values <= 0.0)
        ):
            raise ValueError(
                "budget_sweep.accelerations must contain finite positive values"
            )

    ista = config.get("recon", {}).get("wavelet_ista")
    if ista:
        if int(ista.get("n_iters", 0)) < 1:
            raise ValueError("recon.wavelet_ista.n_iters must be positive")
        if float(ista.get("threshold", 0.0)) < 0.0:
            raise ValueError("recon.wavelet_ista.threshold must be non-negative")

    for section in ("unet_post", "loupe"):
        settings = config.get(section)
        if not settings:
            continue
        if int(settings.get("epochs", 1)) < 1:
            raise ValueError(f"{section}.epochs must be positive")
        if int(settings.get("batch_size", 1)) < 1:
            raise ValueError(f"{section}.batch_size must be positive")
        learning_rate = float(settings.get("lr", 1e-3))
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(f"{section}.lr must be finite and positive")
        if int(settings.get("base_channels", 1)) < 1:
            raise ValueError(f"{section}.base_channels must be positive")

    n_examples = int(config.get("outputs", {}).get("n_examples", 1))
    if n_examples < 1:
        raise ValueError("outputs.n_examples must be positive")


def config_fingerprint(config: dict[str, Any], length: int = 12) -> str:
    """Stable SHA-256 fingerprint of the resolved configuration."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch; return a NumPy generator for local use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def run_dir(config: dict[str, Any]) -> Path:
    """Return runs/<experiment_name>/<config-hash>/ for this exact config."""
    out = Path("runs") / str(config["experiment_name"]) / config_fingerprint(config)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def save_config_snapshot(
    config: dict[str, Any],
    out_dir: Path,
    name: str,
    *,
    cli_args: dict[str, Any] | None = None,
) -> Path:
    """Write a reproducibility manifest for one script invocation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"config_{name}.json"
    payload = {
        "script": name,
        "config_hash": config_fingerprint(config),
        "config": config,
        "cli_args": cli_args or {},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "git_revision": _git_revision(),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path
