"""Frequency-domain masks with explicit point and full-line acquisition budgets."""

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


def line_count_from_sample_budget(shape: tuple[int, int], n_samples: int) -> int:
    """Maximum number of complete Cartesian columns within a point budget.

    A Cartesian readout acquires a complete column, so a point budget that is
    not divisible by the column height cannot be met exactly without changing
    the acquisition model.  The unused remainder is therefore left unspent.
    Budgets smaller than one complete column are invalid for line sampling.
    """
    validate_budget(shape, n_samples)
    n_rows = shape[0]
    n_lines = n_samples // n_rows
    if n_lines < 1:
        raise ValueError(
            f"n_samples={n_samples} cannot acquire one complete Cartesian "
            f"column of {n_rows} samples"
        )
    return n_lines


def validate_line_count(shape: tuple[int, int], n_lines: int) -> None:
    """Validate an explicit complete-column acquisition budget."""
    n_cols = shape[1]
    if not (1 <= n_lines <= n_cols):
        raise ValueError(f"n_lines={n_lines} must be in [1, {n_cols}]")


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


def fill_full_lines(
    shape: tuple[int, int], cols_by_priority: np.ndarray, n_lines: int
) -> np.ndarray:
    """Build a mask containing exactly ``n_lines`` complete Cartesian columns."""
    validate_line_count(shape, n_lines)
    cols = np.asarray(cols_by_priority)
    if cols.ndim != 1:
        raise ValueError("cols_by_priority must be one-dimensional")
    if not np.issubdtype(cols.dtype, np.integer):
        if not np.all(np.isfinite(cols)) or not np.all(cols == np.floor(cols)):
            raise ValueError("column indices must be finite integers")
    cols = cols.astype(np.int64, copy=False)
    if cols.size < n_lines:
        raise ValueError(f"need {n_lines} columns but only {cols.size} were provided")
    if np.any((cols < 0) | (cols >= shape[1])):
        raise ValueError(f"column indices must be in [0, {shape[1] - 1}]")
    if np.unique(cols[:n_lines]).size != n_lines:
        raise ValueError("the selected priority prefix contains duplicate columns")

    mask = np.zeros(shape, dtype=np.float32)
    mask[:, cols[:n_lines]] = 1.0
    return mask


def fill_lines(shape: tuple[int, int], cols_by_priority: np.ndarray, n_samples: int) -> np.ndarray:
    """Compatibility wrapper using a maximum point budget for full columns.

    ``n_samples`` is retained for existing callers.  Only the largest number
    of complete columns that fits within that budget is acquired; no partial
    column is ever emitted.  New code with an acquisition-unit budget should
    call :func:`fill_full_lines` directly.
    """
    n_lines = line_count_from_sample_budget(shape, n_samples)
    return fill_full_lines(shape, cols_by_priority, n_lines)


def equispaced_lines_mask(shape: tuple[int, int], n_samples: int) -> np.ndarray:
    """Complete Cartesian columns on a regular grid within a point budget."""
    n_cols_total = shape[1]
    n_lines = line_count_from_sample_budget(shape, n_samples)
    # Evenly spaced distinct columns; the step W / n_lines is >= 1.
    cols = np.floor(np.arange(n_lines) * (n_cols_total / n_lines)).astype(np.int64)
    # Shift so the column nearest the center lands exactly on it.
    nearest = cols[np.argmin(np.abs(cols - n_cols_total // 2))]
    cols = (cols + (n_cols_total // 2 - nearest)) % n_cols_total
    order = np.argsort(np.abs(cols - n_cols_total // 2), kind="stable")
    return fill_full_lines(shape, cols[order], n_lines)


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
    n_cols_total = shape[1]
    n_lines = line_count_from_sample_budget(shape, n_samples)
    if n_center_lines < 0:
        raise ValueError("n_center_lines must be non-negative")
    n_center_lines = min(n_center_lines, n_lines)

    offsets = np.abs(np.arange(n_cols_total) - n_cols_total // 2)
    center_cols = np.argsort(offsets, kind="stable")[:n_center_lines].astype(np.int64)
    remaining = np.setdiff1d(np.arange(n_cols_total, dtype=np.int64), center_cols)
    n_draw = n_lines - n_center_lines
    if n_draw:
        prob = (1.0 + offsets[remaining] / (0.1 * n_cols_total)) ** (-decay)
        prob /= prob.sum()
        drawn = rng.choice(remaining, size=n_draw, replace=False, p=prob)
    else:
        drawn = np.empty(0, dtype=np.int64)
    # Center columns first, then random draws from nearest to farthest.
    drawn = drawn[np.argsort(offsets[drawn], kind="stable")]
    return fill_full_lines(shape, np.concatenate([center_cols, drawn]), n_lines)


def multilevel_random_mask(
    shape: tuple[int, int],
    n_samples: int,
    rng: np.random.Generator,
    n_levels: int = 4,
    decay: float = 1.5,
    n_center: int = 0,
) -> np.ndarray:
    """Multilevel random sampling over dyadic radial annuli.

    The frequency plane is split into n_levels annuli with dyadic radii; each
    level gets a share of the budget proportional to its area times a
    per-level density 2^(-decay * level), capped at full sampling. Within a
    level, locations are drawn uniformly without replacement. Leftover budget
    cascades outward (and back inward) so the total is met exactly.
    """
    validate_budget(shape, n_samples, n_center)
    if n_levels < 1:
        raise ValueError("n_levels must be positive")
    if not np.isfinite(decay) or decay < 0.0:
        raise ValueError("decay must be finite and non-negative")
    r = radius_map(shape).ravel()
    r_max = float(r.max())
    # Dyadic annulus edges: [0, r_max/2^(L-1), ..., r_max/2, r_max].
    edges = [0.0] + [r_max / 2 ** (n_levels - 1 - l) for l in range(n_levels)]
    level_of = np.digitize(r, edges[1:-1])
    center = center_indices(shape, n_center)
    available = np.ones(r.size, dtype=bool)
    available[center] = False
    sizes = np.bincount(level_of[available], minlength=n_levels)

    density = 2.0 ** (-decay * np.arange(n_levels))
    weights = sizes * density
    remaining_budget = n_samples - center.size
    if remaining_budget == 0:
        return mask_from_indices(shape, center)
    targets = np.floor(remaining_budget * weights / weights.sum()).astype(np.int64)
    targets = np.minimum(targets, sizes)
    # Distribute the remaining budget innermost-first into levels with room.
    shortfall = remaining_budget - int(targets.sum())
    for level in list(range(n_levels)) * 2:
        if shortfall <= 0:
            break
        room = int(sizes[level] - targets[level])
        add = min(room, shortfall)
        targets[level] += add
        shortfall -= add

    chosen = [
        rng.choice(
            np.flatnonzero((level_of == level) & available),
            size=int(targets[level]),
            replace=False,
        )
        for level in range(n_levels)
        if targets[level] > 0
    ]
    return mask_from_indices(shape, np.concatenate([center, *chosen]))


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Jaccard overlap |A and B| / |A or B| between two binary masks."""
    a = mask_a > 0.5
    b = mask_b > 0.5
    union = float(np.logical_or(a, b).sum())
    if union == 0.0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / union
