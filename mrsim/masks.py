"""Baseline frequency-domain measurement mask generators with exact budgets."""

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
    """Distance of each frequency-domain location from the centered DC coefficient."""
    yy = np.arange(shape[0]) - shape[0] // 2
    xx = np.arange(shape[1]) - shape[1] // 2
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    return np.hypot(gy, gx)


def center_indices(shape: tuple[int, int], n_center: int) -> np.ndarray:
    """Flat indices of the n_center locations closest to the frequency-domain center."""
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


def fill_lines(shape: tuple[int, int], cols_by_priority: np.ndarray, n_samples: int) -> np.ndarray:
    """Build a Cartesian column mask from a priority-ordered column list.

    The highest-priority columns are fully sampled; the last needed column is
    filled partially (rows from the center outward) so the measurement budget
    is met exactly.
    """
    validate_budget(shape, n_samples)
    n_rows = shape[0]
    n_full = n_samples // n_rows
    remainder = n_samples - n_full * n_rows
    needed = n_full + (1 if remainder else 0)
    if needed > cols_by_priority.size:
        raise ValueError(f"need {needed} columns but only {cols_by_priority.size} were provided")

    mask = np.zeros(shape, dtype=np.float32)
    mask[:, cols_by_priority[:n_full]] = 1.0
    if remainder > 0:
        rows = np.argsort(np.abs(np.arange(n_rows) - n_rows // 2), kind="stable")[:remainder]
        mask[rows, cols_by_priority[n_full]] = 1.0
    return mask


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
    return fill_lines(shape, cols[order], n_samples)


def variable_density_lines_mask(
    shape: tuple[int, int],
    n_samples: int,
    rng: np.random.Generator,
    decay: float = 2.0,
    n_center_lines: int = 2,
) -> np.ndarray:
    """Random Cartesian columns with polynomially decaying density.

    A block of n_center_lines columns around the frequency-domain center is
    always fully prioritized; the rest are drawn without replacement with
    probability decaying polynomially in the column offset from the center.
    """
    validate_budget(shape, n_samples)
    n_rows, n_cols_total = shape
    needed = int(np.ceil(n_samples / n_rows))
    n_center_lines = min(n_center_lines, needed)

    offsets = np.abs(np.arange(n_cols_total) - n_cols_total // 2)
    center_cols = np.argsort(offsets, kind="stable")[:n_center_lines].astype(np.int64)
    remaining = np.setdiff1d(np.arange(n_cols_total, dtype=np.int64), center_cols)
    prob = (1.0 + offsets[remaining] / (0.1 * n_cols_total)) ** (-decay)
    prob /= prob.sum()
    drawn = rng.choice(remaining, size=needed - n_center_lines, replace=False, p=prob)
    # Center columns first, then random draws from nearest to farthest, so the
    # partial column (if any) lands on the least central random draw.
    drawn = drawn[np.argsort(offsets[drawn], kind="stable")]
    return fill_lines(shape, np.concatenate([center_cols, drawn]), n_samples)


def multilevel_random_mask(
    shape: tuple[int, int],
    n_samples: int,
    rng: np.random.Generator,
    n_levels: int = 4,
    decay: float = 1.5,
) -> np.ndarray:
    """Multilevel random sampling over dyadic radial annuli.

    The frequency plane is split into n_levels annuli with dyadic radii; each
    level gets a share of the budget proportional to its area times a
    per-level density 2^(-decay * level), capped at full sampling. Within a
    level, locations are drawn uniformly without replacement. Leftover budget
    cascades outward (and back inward) so the total is met exactly.
    """
    validate_budget(shape, n_samples)
    r = radius_map(shape).ravel()
    r_max = float(r.max())
    # Dyadic annulus edges: [0, r_max/2^(L-1), ..., r_max/2, r_max].
    edges = [0.0] + [r_max / 2 ** (n_levels - 1 - l) for l in range(n_levels)]
    level_of = np.digitize(r, edges[1:-1])
    sizes = np.bincount(level_of, minlength=n_levels)

    density = 2.0 ** (-decay * np.arange(n_levels))
    weights = sizes * density
    targets = np.floor(n_samples * weights / weights.sum()).astype(np.int64)
    targets = np.minimum(targets, sizes)
    # Distribute the remaining budget innermost-first into levels with room.
    shortfall = n_samples - int(targets.sum())
    for level in list(range(n_levels)) * 2:
        if shortfall <= 0:
            break
        room = int(sizes[level] - targets[level])
        add = min(room, shortfall)
        targets[level] += add
        shortfall -= add

    chosen = [
        rng.choice(np.flatnonzero(level_of == level), size=int(targets[level]), replace=False)
        for level in range(n_levels)
        if targets[level] > 0
    ]
    return mask_from_indices(shape, np.concatenate(chosen))


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Jaccard overlap |A and B| / |A or B| between two binary masks."""
    a = mask_a > 0.5
    b = mask_b > 0.5
    union = float(np.logical_or(a, b).sum())
    if union == 0.0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / union
