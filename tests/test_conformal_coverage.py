"""Split-conformal calibration: marginal coverage, and its limits."""

from __future__ import annotations

import numpy as np
import pytest

from mrsim import certify


def test_quantile_uses_the_finite_sample_correction():
    scores = np.arange(1.0, 11.0)  # n = 10
    # ceil(11 * 0.9) = 10 -> the largest score.
    assert certify.conformal_quantile(scores, 0.1) == 10.0
    # ceil(11 * 0.8) = 9 -> the ninth smallest.
    assert certify.conformal_quantile(scores, 0.2) == 9.0


def test_too_few_calibration_points_give_an_infinite_bound():
    """With n < 1/alpha - 1 no finite threshold is justified; say so."""
    assert certify.conformal_quantile(np.arange(5.0), 0.01) == float("inf")


def test_marginal_coverage_holds_over_repeated_splits():
    """The guarantee is over the joint draw, so it is checked by repetition."""
    rng = np.random.default_rng(0)
    alpha = 0.1
    covered = []
    for _ in range(2000):
        sample = rng.gamma(shape=2.0, scale=1.0, size=41)
        certificate = certify.calibrate(sample[:40], alpha)
        covered.append(bool(certificate.covers(sample[40:])[0]))
    rate = float(np.mean(covered))
    assert rate >= 1.0 - alpha - 0.02, f"under-coverage: {rate}"
    # Conformal is valid but not exact; it over-covers slightly by construction.
    assert rate <= 1.0 - alpha + 0.05


def test_coverage_is_distribution_free_across_score_shapes():
    rng = np.random.default_rng(1)
    alpha = 0.2
    for draw in (
        lambda n: rng.standard_exponential(n),
        lambda n: rng.pareto(2.0, n),
        lambda n: np.abs(rng.standard_cauchy(n)),
    ):
        covered = []
        for _ in range(1500):
            sample = draw(51)
            certificate = certify.calibrate(sample[:50], alpha)
            covered.append(bool(certificate.covers(sample[50:])[0]))
        assert float(np.mean(covered)) >= 1.0 - alpha - 0.03


def test_exchangeability_is_required_and_shift_breaks_it():
    """Under prior shift the guarantee genuinely fails; this is not pessimism."""
    rng = np.random.default_rng(2)
    calibration = rng.standard_exponential(200)
    shifted = rng.standard_exponential(2000) * 4.0
    certificate = certify.calibrate(calibration, 0.1)
    assert certificate.empirical_coverage(shifted) < 0.9


def test_error_scores_shape_and_validation():
    recon = np.zeros((4, 8, 8), dtype=np.complex128)
    truth = np.ones((4, 8, 8))
    scores = certify.error_scores(recon, truth)
    assert scores.shape == (4,)
    assert np.allclose(scores, 8.0)
    with pytest.raises(ValueError):
        certify.error_scores(recon, np.ones((3, 8, 8)))


def test_coverage_interval_widens_as_the_split_shrinks():
    wide = certify.coverage_interval(12, 0.1)
    narrow = certify.coverage_interval(500, 0.1)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_calibrate_rejects_bad_alpha():
    with pytest.raises(ValueError):
        certify.calibrate(np.arange(10.0), 0.0)
    with pytest.raises(ValueError):
        certify.calibrate(np.arange(10.0), 1.0)
