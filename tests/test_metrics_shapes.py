import math

import numpy as np
import torch

from mrsim.artifacts import aliasing_energy_ratio, expected_zero_filled_mse, psf_metrics
from mrsim.masks import uniform_random_mask
from mrsim.metrics import evaluate


def test_evaluate_returns_finite_scalars():
    rng = np.random.default_rng(0)
    truth = rng.random((16, 16))
    recon = truth + 0.05 * rng.standard_normal((16, 16))
    result = evaluate(recon, truth)
    assert set(result) == {"mse", "psnr", "ssim", "nrmse"}
    for key, value in result.items():
        assert isinstance(value, float), key
        assert math.isfinite(value), key


def test_evaluate_identical_inputs_finite():
    truth = np.random.default_rng(1).random((16, 16))
    result = evaluate(truth.copy(), truth)
    assert all(math.isfinite(v) for v in result.values())
    assert result["mse"] == 0.0


def test_artifact_metrics_finite():
    mask = uniform_random_mask((16, 16), 64, np.random.default_rng(0))
    image = torch.rand(16, 16, generator=torch.Generator().manual_seed(0))

    ratio = aliasing_energy_ratio(image, mask)
    assert isinstance(ratio, float)
    assert 0.0 <= ratio <= 1.0

    psf = psf_metrics(mask)
    assert all(isinstance(v, float) and math.isfinite(v) for v in psf.values())

    power = np.random.default_rng(2).random((16, 16))
    score = expected_zero_filled_mse(mask, power)
    assert isinstance(score, float) and math.isfinite(score)
