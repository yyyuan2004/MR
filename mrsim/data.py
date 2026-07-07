"""Synthetic test signal generation."""

from __future__ import annotations

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
