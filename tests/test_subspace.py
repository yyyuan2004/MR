import numpy as np
import torch

from mrsim.artifacts import decompose_error, subspace_nullspace_leakage
from mrsim.data import random_ellipse_phantom
from mrsim.greedy import greedy_subspace_aoptimal
from mrsim.masks import variable_density_mask
from mrsim.recon import generative_recon, simulate_measurements, subspace_recon
from mrsim.subspace import fit_subspace, generator_jacobian_basis, to_kspace_basis

SIZE = 16
D = 6


def _train_images(n: int = 24) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack([random_ellipse_phantom(SIZE, rng) for _ in range(n)])


def _basis() -> np.ndarray:
    basis, _ = fit_subspace(_train_images(), D)
    return basis


def test_fit_subspace_orthonormal_and_energy():
    basis, energy_ratio = fit_subspace(_train_images(), D)
    assert basis.shape == (SIZE * SIZE, D)
    assert np.allclose(basis.conj().T @ basis, np.eye(D), atol=1e-10)
    assert 0.0 < energy_ratio <= 1.0


def test_kspace_basis_preserves_column_norms():
    # F is unitary, so Phi = F B has the same column norms as B.
    basis = _basis()
    phi = to_kspace_basis(basis, (SIZE, SIZE))
    assert np.allclose(
        np.linalg.norm(phi, axis=0), np.linalg.norm(basis, axis=0), atol=1e-6
    )


def test_subspace_recon_fills_nullspace():
    basis = _basis()
    truth = torch.from_numpy(random_ellipse_phantom(SIZE, np.random.default_rng(99)))
    mask = variable_density_mask((SIZE, SIZE), 64, np.random.default_rng(1), n_center=5)
    y = simulate_measurements(truth[None], mask, noise_std=0.0)[0]
    recon = subspace_recon(y, mask, basis, lam=1e-6)
    dec = decompose_error(recon, truth, mask)
    truth_norm = torch.linalg.vector_norm(truth.to(torch.complex64)).item()
    assert dec.recon_nullspace_norm > 1e-4 * truth_norm
    assert dec.no_nullspace_content is False


def test_trace_monotonically_decreases():
    phi = to_kspace_basis(_basis(), (SIZE, SIZE))
    _, history = greedy_subspace_aoptimal(
        phi, 40, sigma2=1e-3, n_center=5, return_trace=True
    )
    assert len(history) == 40 - 5 + 1
    for previous, current in zip(history, history[1:]):
        assert current <= previous + 1e-12


def test_budget_and_center_enforced():
    phi = to_kspace_basis(_basis(), (SIZE, SIZE))
    mask = greedy_subspace_aoptimal(phi, 40, n_center=5)
    assert mask.shape == (SIZE, SIZE)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert int(mask.sum()) == 40
    assert mask[SIZE // 2, SIZE // 2] == 1.0


def _brute_force_aoptimal(phi: np.ndarray, n_samples: int, ridge: float) -> np.ndarray:
    """Reference greedy: recompute trace((G + ridge I)^-1) for every candidate."""
    n_locations, d = phi.shape
    selected = np.zeros(n_locations, dtype=bool)
    gram = np.zeros((d, d), dtype=np.complex128)
    for _ in range(n_samples):
        best, best_trace = -1, np.inf
        for k in range(n_locations):
            if selected[k]:
                continue
            trial = gram + np.outer(phi[k].conj(), phi[k])
            trace = float(np.trace(np.linalg.inv(trial + ridge * np.eye(d))).real)
            if trace < best_trace:
                best, best_trace = k, trace
        selected[best] = True
        gram += np.outer(phi[best].conj(), phi[best])
    return selected


def test_beta_zero_matches_brute_force_a_optimal():
    rng = np.random.default_rng(3)
    phi = rng.standard_normal((64, 4)) + 1j * rng.standard_normal((64, 4))
    ridge = 1e-6
    mask = greedy_subspace_aoptimal(phi, 12, shape=(8, 8), ridge=ridge, beta=0.0)
    expected = _brute_force_aoptimal(phi, 12, ridge)
    assert np.array_equal(mask.ravel() > 0.5, expected)


def test_artifact_aware_beta_changes_only_scoring():
    # beta > 0 must still meet the budget; beta = 0 is the pure path above.
    phi = to_kspace_basis(_basis(), (SIZE, SIZE))
    mask = greedy_subspace_aoptimal(phi, 30, n_center=3, beta=1.0, n_candidates=8)
    assert int(mask.sum()) == 30


def test_subspace_leakage_bounds_and_ordering():
    basis = _basis()
    empty = np.zeros((SIZE, SIZE), dtype=np.float32)
    full = np.ones((SIZE, SIZE), dtype=np.float32)
    partial = variable_density_mask((SIZE, SIZE), 64, np.random.default_rng(2))
    assert np.isclose(subspace_nullspace_leakage(basis, full), 0.0, atol=1e-10)
    assert np.isclose(subspace_nullspace_leakage(basis, empty), 1.0, atol=1e-10)
    leak = subspace_nullspace_leakage(basis, partial)
    assert 0.0 < leak < 1.0


def test_generator_jacobian_recovers_linear_basis():
    # For a linear generator G(z) = (U z), the Jacobian *is* U: the projectors
    # onto the two column spans must coincide (QR leaves the span unchanged).
    basis = _basis()
    weight = torch.from_numpy(basis.astype(np.float32))

    def linear_generator(z: torch.Tensor) -> torch.Tensor:
        return (weight @ z).reshape(SIZE, SIZE)

    jac_basis = generator_jacobian_basis(linear_generator, torch.zeros(D))
    proj_jac = jac_basis @ jac_basis.conj().T
    proj_ref = basis @ basis.conj().T
    assert np.allclose(proj_jac, proj_ref, atol=1e-4)


def test_generative_recon_reduces_residual():
    basis = _basis()
    weight = torch.from_numpy(basis.astype(np.float32))

    def linear_generator(z: torch.Tensor) -> torch.Tensor:
        return (weight @ z).reshape(SIZE, SIZE)

    torch.manual_seed(0)
    z_true = torch.randn(D)
    truth = linear_generator(z_true)
    mask = variable_density_mask((SIZE, SIZE), 64, np.random.default_rng(4), n_center=5)
    y = simulate_measurements(truth[None], mask, noise_std=0.0)[0]

    recon = generative_recon(y, mask, linear_generator, torch.zeros(D), steps=150, lr=0.1)
    final_error = torch.linalg.vector_norm(recon - truth.to(torch.complex64)).item()
    init_error = torch.linalg.vector_norm(truth.to(torch.complex64)).item()
    assert final_error < 0.2 * init_error
