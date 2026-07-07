import numpy as np
import torch

from mrsim.artifacts import decompose_error
from mrsim.data import random_ellipse_phantom
from mrsim.experiment import greedy_noise_var, mean_power_spectrum
from mrsim.masks import variable_density_mask
from mrsim.recon import ridge, simulate_measurements, zero_filled

SIZE = 64


def _setup() -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(0)
    images = torch.from_numpy(
        np.stack([random_ellipse_phantom(SIZE, rng) for _ in range(8)])
    )
    truth = images[0]
    spectrum = mean_power_spectrum(images[1:])  # estimated on separate signals
    mask = variable_density_mask(
        (SIZE, SIZE), (SIZE * SIZE) // 4, np.random.default_rng(1), n_center=20
    )
    y = simulate_measurements(truth[None], mask, noise_std=0.005,
                              generator=torch.Generator().manual_seed(0))[0]
    return truth, mask, y, spectrum


def test_linear_methods_are_range_confined():
    truth, mask, y, spectrum = _setup()
    truth_norm = torch.linalg.vector_norm(truth.to(torch.complex64)).item()
    for recon in (zero_filled(y), ridge(y, mask, 0.005**2, spectrum=spectrum)):
        dec = decompose_error(recon, truth, mask)
        assert dec.recon_nullspace_norm <= 1e-4 * truth_norm
        assert dec.no_nullspace_content is True


def test_wiener_is_not_scalar_multiple_of_zero_filled():
    # Regression guard: with a spectrum, the shrinkage must vary per frequency
    # coefficient. A regularization weight comparable to the weakest spectrum
    # entries makes the variation clearly visible; if ridge ignored the
    # spectrum (degenerate scalar behavior), the residual below would be ~0
    # for every lam.
    truth, mask, y, spectrum = _setup()
    zf = zero_filled(y)
    wiener = ridge(y, mask, 0.01, spectrum=spectrum)
    # Best scalar fit a = <zf, w> / <zf, zf>; the residual must be substantial.
    a = torch.sum(zf.conj() * wiener) / torch.sum(zf.conj() * zf)
    residual = torch.linalg.vector_norm(wiener - a * zf).item()
    assert residual > 1e-2 * torch.linalg.vector_norm(wiener).item()


def test_greedy_noise_var_matches_measurement_noise():
    cfg = {"measurement": {"noise_std": 0.005}, "greedy": {}}
    assert greedy_noise_var(cfg) == 0.005**2
    cfg_override = {"measurement": {"noise_std": 0.005}, "greedy": {"noise_var": 0.5}}
    assert greedy_noise_var(cfg_override) == 0.5
    assert greedy_noise_var({}) == 0.0
