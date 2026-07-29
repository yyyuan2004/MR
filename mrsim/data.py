"""Synthetic test signal generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, rotate


def _coordinate_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-1.0, 1.0, size, endpoint=False)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return yy, xx


def _ellipse(
    yy: np.ndarray,
    xx: np.ndarray,
    cy: float,
    cx: float,
    a: float,
    b: float,
    theta: float,
) -> np.ndarray:
    # Rotate coordinates into the ellipse frame before the axis test.
    c, s = np.cos(theta), np.sin(theta)
    xr = (xx - cx) * c + (yy - cy) * s
    yr = -(xx - cx) * s + (yy - cy) * c
    return ((xr / a) ** 2 + (yr / b) ** 2 <= 1.0).astype(np.float64)


def random_ellipse_phantom(
    size: int,
    rng: np.random.Generator,
    min_ellipses: int = 3,
    max_ellipses: int = 8,
) -> np.ndarray:
    """Random superposition of ellipses, clipped to [0, 1]."""
    yy, xx = _coordinate_grid(size)
    image = np.zeros((size, size), dtype=np.float64)

    # Large background ellipse so every test signal has a compact support.
    image += _ellipse(
        yy,
        xx,
        0.0,
        0.0,
        float(rng.uniform(0.7, 0.9)),
        float(rng.uniform(0.7, 0.9)),
        float(rng.uniform(0.0, np.pi)),
    ) * float(rng.uniform(0.2, 0.4))

    n = int(rng.integers(min_ellipses, max_ellipses + 1))
    for _ in range(n):
        cy, cx = rng.uniform(-0.5, 0.5, size=2)
        a, b = rng.uniform(0.05, 0.4, size=2)
        theta = float(rng.uniform(0.0, np.pi))
        image += _ellipse(yy, xx, float(cy), float(cx), float(a), float(b), theta) * float(
            rng.uniform(-0.4, 0.8)
        )
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def shepp_logan(size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Classical Shepp-Logan test image resized to `size`, optionally randomly rotated."""
    image = shepp_logan_phantom()
    if rng is not None:
        image = rotate(image, angle=float(rng.uniform(-15.0, 15.0)), mode="constant")
    image = resize(image, (size, size), anti_aliasing=True)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _find_image_tensor(obj):
    """Locate the image stack inside an arbitrarily wrapped torch payload."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in ("images", "data", "x", "volume", "slices"):
            if key in obj:
                return _find_image_tensor(obj[key])
        for value in obj.values():
            found = _find_image_tensor(value)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)) and obj:
        return _find_image_tensor(obj[0])
    return None


def load_array_dataset(path, n_images: int, size: int) -> torch.Tensor:
    """Load an (N, S, S) real stack in [0, 1] from a .npy or .pt file.

    Accepts a bare tensor/array or a dict wrapping one under a common key, so an
    existing dataset.pt can be reused in place without conversion. Any metadata
    in the payload is ignored: the caller supplies its own.

    Split leakage warning: train_val_test_split slices sequentially and does not
    shuffle, so an external stack must be grouped by subject before saving —
    all slices from one subject in a single contiguous block. Interleaved
    stacking puts the same subject in train and test, leaking the evaluation
    split into the fitted prior spectrum.
    """
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix in (".pt", ".pth"):
        try:
            payload = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            payload = torch.load(p, map_location="cpu", weights_only=False)
        tensor = _find_image_tensor(payload)
        if tensor is None:
            raise ValueError(f"no image tensor found inside {p}")
        arr = tensor.detach().cpu().numpy()
    else:
        raise ValueError(f"unsupported dataset extension {p.suffix!r}; use .npy or .pt")

    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 1:      # (N, 1, S, S) -> (N, S, S)
        arr = arr[:, 0]
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"expected (N, S, S) with square images, got {arr.shape}")
    if arr.shape[1] != size:
        raise ValueError(f"config image_size={size} but file has {arr.shape[1]}")
    if arr.shape[0] < n_images:
        raise ValueError(f"config n_images={n_images} but file has {arr.shape[0]}")

    arr = np.ascontiguousarray(arr[:n_images], dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if lo < -1e-6 or hi > 1.0 + 1e-6:
        raise ValueError(
            f"values must lie in [0, 1], got [{lo:.4f}, {hi:.4f}]; rescale the file first"
        )
    return torch.from_numpy(arr)


def generate_dataset(
    n_images: int,
    size: int,
    seed: int,
    phantom: str = "ellipses",
    min_ellipses: int = 3,
    max_ellipses: int = 8,
) -> torch.Tensor:
    """Generate a stack of synthetic test images with shape (n_images, size, size)."""
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(n_images):
        if phantom == "ellipses":
            images.append(random_ellipse_phantom(size, rng, min_ellipses, max_ellipses))
        elif phantom == "shepp_logan":
            images.append(shepp_logan(size, rng))
        else:
            raise ValueError(f"unknown phantom type: {phantom!r}")
    return torch.from_numpy(np.stack(images))
