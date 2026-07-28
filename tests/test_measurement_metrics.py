import numpy as np
import pytest
import torch

from mrsim.artifacts import (
    decompose_error,
    expected_zero_filled_mse,
    expected_zero_filled_noise_mse,
    measurement_residual,
    measurement_residual_norm,
)
from mrsim.fft_ops import as_mask_tensor, fft2c
from mrsim.metrics import complex_mse, magnitude_mse, mse
from mrsim.recon import (
    ridge,
    sample_complex_noise_like,
    simulate_measurements,
    zero_filled,
)


def _masks() -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((8, 8), dtype=np.float32)
    left[:, :5] = 1.0
    top = np.zeros((8, 8), dtype=np.float32)
    top[:5, :] = 1.0
    return left, top


def test_explicit_full_kspace_noise_is_reused_across_masks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.arange(2 * 8 * 8, dtype=torch.float64, device=device).reshape(2, 8, 8)
    noise = torch.full(
        images.shape,
        0.125 + 0.25j,
        dtype=torch.complex128,
        device=device,
    )
    left, top = _masks()

    y_left = simulate_measurements(images, left, noise=noise)
    y_top = simulate_measurements(images, top, noise=noise)
    full_noisy_kspace = fft2c(images.to(torch.complex128)) + noise

    assert y_left.device == images.device
    assert y_left.dtype == torch.complex128
    assert torch.allclose(y_left, as_mask_tensor(left, like=full_noisy_kspace) * full_noisy_kspace)
    assert torch.allclose(y_top, as_mask_tensor(top, like=full_noisy_kspace) * full_noisy_kspace)
    overlap = as_mask_tensor(left * top, like=full_noisy_kspace).bool()
    assert torch.equal(y_left[:, overlap], y_top[:, overlap])


def test_sampled_noise_follows_reference_device_dtype_and_variance_convention():
    reference = torch.zeros((128, 128), dtype=torch.float64)
    generator = torch.Generator().manual_seed(7)
    noise = sample_complex_noise_like(reference, 0.2, generator=generator)

    assert noise.device == reference.device
    assert noise.dtype == torch.complex128
    empirical_power = float((noise.abs() ** 2).mean())
    assert empirical_power == pytest.approx(0.2**2, rel=0.08)


def test_internal_and_explicit_noise_paths_agree_for_the_same_seed():
    images = torch.zeros((2, 8, 8), dtype=torch.float32)
    mask, _ = _masks()
    direct_generator = torch.Generator().manual_seed(17)
    explicit_generator = torch.Generator().manual_seed(17)

    direct = simulate_measurements(
        images,
        mask,
        noise_std=0.15,
        generator=direct_generator,
    )
    noise = sample_complex_noise_like(
        images,
        0.15,
        generator=explicit_generator,
    )
    explicit = simulate_measurements(images, mask, noise=noise)

    assert torch.equal(direct, explicit)


def test_explicit_noise_rejects_ambiguous_or_non_full_grid_inputs():
    images = torch.zeros((8, 8))
    noise = torch.zeros((8, 8), dtype=torch.complex64)
    mask, _ = _masks()

    with pytest.raises(ValueError, match="mutually exclusive"):
        simulate_measurements(images, mask, noise_std=0.1, noise=noise)
    with pytest.raises(ValueError, match="full k-space shape"):
        simulate_measurements(images, mask, noise=noise[:4])
    with pytest.raises(TypeError, match="complex"):
        simulate_measurements(images, mask, noise=noise.real)


def test_measurement_residual_is_test_time_computable_and_not_oracle_error():
    generator = torch.Generator().manual_seed(11)
    truth = torch.randn((8, 8), generator=generator)
    mask, _ = _masks()
    noise = sample_complex_noise_like(truth, 0.1, generator=generator)
    measurements = simulate_measurements(truth, mask, noise=noise)
    recon = zero_filled(measurements)

    residual = measurement_residual(recon, measurements, mask)
    decomposition = decompose_error(recon, truth, mask)

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)
    assert measurement_residual_norm(recon, measurements, mask) == pytest.approx(0.0, abs=1e-6)
    assert decomposition.observed_subspace_error_norm > 0.0
    assert torch.equal(
        decomposition.observed_subspace_error,
        decomposition.consistency_error,
    )
    assert decomposition.observed_subspace_error_norm == decomposition.consistency_norm


def test_measurement_residual_has_y_minus_a_recon_sign():
    mask, _ = _masks()
    measurements = torch.zeros((8, 8), dtype=torch.complex64)
    measurements[2, 3] = 2.0 - 1.0j

    residual = measurement_residual(torch.zeros((8, 8)), measurements, mask)

    assert torch.equal(residual, measurements)
    assert measurement_residual_norm(torch.zeros((8, 8)), measurements, mask) == pytest.approx(
        np.sqrt(5.0)
    )


def test_complex_and_magnitude_mse_have_distinct_explicit_semantics():
    recon = np.array([1j, -1.0 + 0.0j])
    truth = np.array([1.0 + 0.0j, 1.0 + 0.0j])

    assert magnitude_mse(recon, truth) == 0.0
    assert complex_mse(recon, truth) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="complex_mse or magnitude_mse"):
        mse(recon, truth)


def test_ridge_prior_mean_fills_unmeasured_frequencies():
    mask, _ = _masks()
    mask_t = torch.from_numpy(mask).to(torch.float64)
    measurements = torch.complex(
        2.0 * mask_t,
        -0.5 * mask_t,
    )
    prior_mean = torch.complex(
        torch.arange(64, dtype=torch.float64).reshape(8, 8) / 64.0,
        torch.full((8, 8), 0.25, dtype=torch.float64),
    )
    lam = 0.5

    reconstruction = ridge(
        measurements,
        mask,
        lam,
        prior_mean=prior_mean,
    )
    reconstructed_kspace = fft2c(reconstruction)
    weight = 1.0 / (1.0 + lam)
    expected = prior_mean + mask_t * weight * (measurements - prior_mean)

    assert reconstructed_kspace.dtype == torch.complex128
    assert torch.allclose(reconstructed_kspace, expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        reconstructed_kspace[mask_t == 0],
        prior_mean[mask_t == 0],
        atol=1e-10,
        rtol=1e-10,
    )


def test_ridge_without_prior_mean_preserves_zero_mean_behavior():
    mask, _ = _masks()
    mask_t = torch.from_numpy(mask)
    measurements = torch.complex(2.0 * mask_t, -0.5 * mask_t)
    lam = 0.5

    reconstructed_kspace = fft2c(ridge(measurements, mask, lam))

    assert torch.allclose(
        reconstructed_kspace,
        mask_t * (1.0 / (1.0 + lam)) * measurements,
        atol=1e-6,
    )


def test_noiseless_wiener_handles_zero_prior_variance_without_nan():
    mask, _ = _masks()
    measurements = torch.ones((8, 8), dtype=torch.complex64) * torch.from_numpy(mask)
    spectrum = np.ones((8, 8), dtype=np.float64)
    spectrum[0, 0] = 0.0

    reconstruction = ridge(measurements, mask, 0.0, spectrum=spectrum)

    assert torch.isfinite(reconstruction.real).all()
    assert torch.isfinite(reconstruction.imag).all()


def test_noisy_zero_filled_score_adds_sampled_noise_term():
    mask, _ = _masks()
    mean_power = np.ones_like(mask)
    noise_std = 0.2
    expected_noise = noise_std**2 * mask.mean()

    assert expected_zero_filled_noise_mse(mask, noise_std) == pytest.approx(expected_noise)
    assert expected_zero_filled_mse(
        mask,
        mean_power,
        noise_std=noise_std,
    ) == pytest.approx((1.0 - mask).mean() + expected_noise)
