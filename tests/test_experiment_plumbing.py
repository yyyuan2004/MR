from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from mrsim import experiment
from mrsim.config import config_fingerprint, validate_config


def _config() -> dict:
    return {
        "experiment_name": "unit",
        "seed": 13,
        "data": {
            "n_images": 8,
            "image_size": 8,
            "n_train": 4,
            "n_val": 2,
            "n_test": 2,
        },
        "measurement": {"noise_std": 0.1},
        "mask": {
            "sampling_fraction": 0.25,
            "center_fraction": 0.0,
            "types": ["uniform_random"],
        },
        "recon": {"ridge_lambda": 0.05},
        "outputs": {"n_examples": 1},
    }


def test_config_fingerprint_is_order_independent_and_split_is_disjoint():
    cfg = _config()
    reordered = dict(reversed(list(cfg.items())))
    assert config_fingerprint(cfg) == config_fingerprint(reordered)

    images = torch.arange(8 * 8 * 8).reshape(8, 8, 8)
    train, validation, test = experiment.train_validation_test_split(images, cfg)
    assert train[:, 0, 0].tolist() == [0, 64, 128, 192]
    assert validation[:, 0, 0].tolist() == [256, 320]
    assert test[:, 0, 0].tolist() == [384, 448]


def test_config_validation_rejects_overlapping_declared_split():
    cfg = _config()
    cfg["data"]["n_test"] = 3
    with pytest.raises(ValueError, match="exceeds"):
        validate_config(cfg)


def test_config_validation_rejects_zero_training_epochs():
    cfg = _config()
    cfg["unet_post"] = {"epochs": 0, "batch_size": 2, "lr": 1e-3}
    with pytest.raises(ValueError, match="unet_post.epochs"):
        validate_config(cfg)


def test_config_validation_rejects_empty_mask_list():
    cfg = _config()
    cfg["mask"]["types"] = []
    with pytest.raises(ValueError, match="mask.types"):
        validate_config(cfg)


@pytest.mark.parametrize("accelerations", [[], [0], [-2], [float("nan")]])
def test_config_validation_rejects_invalid_budget_sweep(accelerations):
    cfg = _config()
    cfg["budget_sweep"] = {"accelerations": accelerations}
    with pytest.raises(ValueError, match="budget_sweep.accelerations"):
        validate_config(cfg)


def test_dataset_cache_rejects_changed_generation_metadata(tmp_path):
    cfg = _config()
    images = experiment.load_or_generate_dataset(cfg, tmp_path)
    assert images.shape == (8, 8, 8)

    changed = copy.deepcopy(cfg)
    changed["data"]["phantom"] = "shepp_logan"
    with pytest.raises(ValueError, match="different data configuration"):
        experiment.load_or_generate_dataset(changed, tmp_path)


def test_random_mask_stream_is_independent_of_requested_order():
    cfg = _config()
    images = torch.rand((4, 8, 8), generator=torch.Generator().manual_seed(3))
    names = ["uniform_random", "variable_density"]

    forward = experiment.build_masks(
        names, cfg, images, np.random.default_rng(99)
    )
    reverse = experiment.build_masks(
        list(reversed(names)), cfg, images, np.random.default_rng(0)
    )

    for name in names:
        assert np.array_equal(forward[name], reverse[name])


def test_mask_evaluation_is_paired_and_order_independent(tmp_path):
    cfg = _config()
    truth = torch.rand((2, 8, 8), generator=torch.Generator().manual_seed(9))
    left = np.zeros((8, 8), dtype=np.float32)
    left[:, :4] = 1.0
    top = np.zeros((8, 8), dtype=np.float32)
    top[:4, :] = 1.0
    masks = {"left": left, "top": top}
    prior_mean, prior_variance, _ = experiment.frequency_prior_statistics(truth)

    first = experiment.evaluate_masks(
        masks,
        truth,
        cfg,
        tmp_path / "first",
        prefix="paired",
        spectrum=prior_variance,
        prior_mean=prior_mean,
        write_examples=False,
    )
    second = experiment.evaluate_masks(
        dict(reversed(list(masks.items()))),
        truth,
        cfg,
        tmp_path / "second",
        prefix="paired",
        spectrum=prior_variance,
        prior_mean=prior_mean,
        write_examples=False,
    )

    columns = [
        "mask",
        "method",
        "image_index",
        "complex_mse",
        "magnitude_mse",
        "measurement_residual_norm",
    ]
    sort_by = ["mask", "method", "image_index"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values(sort_by).reset_index(drop=True),
        second[columns].sort_values(sort_by).reset_index(drop=True),
    )
    assert "oracle_observed_subspace_error_norm" in first
    assert "consistency_norm" not in first
    assert "mse" not in first


def test_mask_evaluation_rejects_wrong_spatial_shape(tmp_path):
    cfg = _config()
    truth = torch.zeros((2, 8, 8))
    wrong = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match test image"):
        experiment.evaluate_masks(
            {"wrong": wrong},
            truth,
            cfg,
            tmp_path,
            prefix="bad_shape",
            write_examples=False,
        )
