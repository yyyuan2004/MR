import numpy as np
import torch

from mrsim.artifacts import decompose
from mrsim.masks import variable_density_mask


def test_components_sum_to_input():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 16, 16, generator=g)
    mask = variable_density_mask((16, 16), 80, np.random.default_rng(0))
    parts = decompose(x, mask)
    total = parts["kept"] + parts["missing"]
    assert torch.allclose(total, x.to(torch.complex64), atol=1e-5)


def test_energy_is_preserved():
    # P is an orthogonal projection, so ||x||^2 = ||Px||^2 + ||(I-P)x||^2.
    g = torch.Generator().manual_seed(1)
    x = torch.randn(16, 16, generator=g)
    mask = variable_density_mask((16, 16), 64, np.random.default_rng(1))
    parts = decompose(x, mask)
    total_energy = float((x**2).sum())
    kept_energy = float((parts["kept"].abs() ** 2).sum())
    missing_energy = float((parts["missing"].abs() ** 2).sum())
    assert np.isclose(kept_energy + missing_energy, total_energy, rtol=1e-4)
