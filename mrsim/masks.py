"""Baseline k-space sampling mask generators with exact sample budgets."""

from __future__ import annotations

import numpy as np


def budget_from_fraction(shape: tuple[int, int], fraction: float) -> int:
    """Number of sampled locations for a given sampling fraction."""
    total = shape[0] * shape[1]
    return max(1, min(total, int(round(fraction * total))))


def validate_budget(shape: tuple[int, int], n_samples: int, n_center: int = 0) -> None:
    total = shape[0] * shape[1]
    if not (1 <= n_samples <= total):
        raise ValueError(f"n_samples={n_samples} must be in [1, {total}]")
    if not (0 <= n_center <= n_samples):
        raise ValueError(f"n_center={n_center} must be in [0, n_samples={n_samples}]")


def radius_map(shape: tuple[int, int]) -> np.ndarray:
    """Distance of each k-space location from the centered DC coefficient."""
    yy = np.arange(shape[0]) - shape[0] // 2
    xx = np.arange(shape[1]) - shape[1] // 2
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    return np.hypot(gy, gx)


def center_indices(shape: tuple[int, int], n_center: int) -> np.ndarray:
    """Flat indices of the n_center locations closest to the k-space center."""
    if n_center <= 0:
        return np.empty(0, dtype=np.int64)
    r = radius_map(shape).ravel()
    # Stable sort gives deterministic tie-breaking by flat index.
    return np.argsort(r, kind="stable")[:n_center].astype(np.int64)


def mask_from_indices(shape: tuple[int, int], indices: np.ndarray) -> np.ndarray:
    """Binary float32 mask with ones at the given flat indices."""
    flat = np.zeros(shape[0] * shape[1], dtype=np.float32)
    flat[np.asarray(indices, dtype=np.int64)] = 1.0
    return flat.reshape(shape)


def uniform_random_mask(
    shape: tuple[int, int],
    n_samples: int,
    rng: np.random.Generator,
    n_center: int = 0,
) -> np.ndarray:
    """Uniformly random selection, optionally forcing a fully sampled center."""
    validate_budget(shape, n_samples, n_center)
    center = center_indices(shape, n_center)
    remaining = np.setdiff1d(np.arange(shape[0] * shape[1], dtype=np.int64), center)
    chosen = rng.choice(remaining, size=n_samples - center.size, replace=False)
    return mask_from_indices(shape, np.concatenate([center, chosen]))


def variable_density_mask(
    shape: tuple[int, int],
    n_samples: int,
    rng: np.random.Generator,
    decay: float = 3.0,
    n_center: int = 0,
) -> np.ndarray:
    """Random selection with polynomially decaying density away from the center."""
    validate_budget(shape, n_samples, n_center)
    center = center_indices(shape, n_center)
    r = radius_map(shape).ravel()
    # Polynomial decay in |k| concentrates samples at low frequencies.
    prob = (1.0 + r / (0.05 * max(shape))) ** (-decay)
    prob[center] = 0.0
    prob /= prob.sum()
    chosen = rng.choice(r.size, size=n_samples - center.size, replace=False, p=prob)
    return mask_from_indices(shape, np.concatenate([center, chosen]))


def equispaced_lines_mask(shape: tuple[int, int], n_samples: int) -> np.ndarray:
    """Fully sampled columns on a regular grid.

    A final partial column (filled from the center row outward) absorbs the
    remainder so the sample budget is met exactly.
    """
    validate_budget(shape, n_samples)
    n_rows, n_cols_total = shape
    n_cols = int(np.ceil(n_samples / n_rows))
    # Evenly spaced distinct columns; the step W / n_cols is >= 1.
    cols = np.floor(np.arange(n_cols) * (n_cols_total / n_cols)).astype(np.int64)
    # Shift so the column nearest the center lands exactly on it.
    nearest = cols[np.argmin(np.abs(cols - n_cols_total // 2))]
    cols = (cols + (n_cols_total // 2 - nearest)) % n_cols_total
    order = np.argsort(np.abs(cols - n_cols_total // 2), kind="stable")

    mask = np.zeros(shape, dtype=np.float32)
    n_full = n_samples // n_rows
    mask[:, cols[order[:n_full]]] = 1.0
    remainder = n_samples - n_full * n_rows
    if remainder > 0:
        partial_col = cols[order[n_full]]
        rows = np.argsort(np.abs(np.arange(n_rows) - n_rows // 2), kind="stable")[:remainder]
        mask[rows, partial_col] = 1.0
    return mask
