"""
Unit Tests for 3D Rotation Augmentation using Pytest.

Tests include:
1. Identity: Rotation by 0 should not change data.
2. Reversibility: Rotating by theta then -theta should return initial data.
3. Batch consistency: Different batch sizes and shapes.

Author: nakmuaycoder
Date: 2026/04
"""

import numpy as np
import pytest
import torch

from ml.src.augmentation import rotate_batch_3d


def test_rotation_identity():
    """Test that 0 degree rotation doesn't change data."""
    x = torch.randn(4, 3, 10)
    ux, uy, uz = torch.zeros(4), torch.zeros(4), torch.ones(4)  # Z-axis
    angles = torch.zeros(4)

    rotated = rotate_batch_3d(x, ux, uy, uz, angles)
    torch.testing.assert_close(x, rotated, atol=1e-6, rtol=1e-6)


def test_rotation_reversibility():
    """Test that rotating by theta then -theta returns to original."""
    batch_size = 32
    x = torch.randn(batch_size, 3, 100)

    # 1. Random axes and random angles
    phi = torch.rand(batch_size) * 2 * np.pi
    costheta = torch.rand(batch_size) * 2 - 1
    theta = torch.acos(costheta)
    ux = torch.sin(theta) * torch.cos(phi)
    uy = torch.sin(theta) * torch.sin(phi)
    uz = torch.cos(theta)

    angles = torch.rand(batch_size) * 3.0  # random angle up to ~170 deg

    # 2. Rotate Forward
    rotated = rotate_batch_3d(x, ux, uy, uz, angles)

    # 3. Rotate Backward (negative angle)
    recovered = rotate_batch_3d(rotated, ux, uy, uz, -angles)

    # 4. Assert
    torch.testing.assert_close(x, recovered, atol=1e-5, rtol=1e-5)


def test_batch_broadcasting():
    """Test single sample vs batch processing."""
    x_single = torch.randn(3, 10)
    ux, uy, uz = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0])
    angle = torch.tensor([np.pi / 2])  # 90 deg

    rotated = rotate_batch_3d(x_single, ux, uy, uz, angle)

    assert rotated.shape == (3, 10)
    # Check that X became -Y (standard 90 deg Z rotation)
    torch.testing.assert_close(rotated[0], -x_single[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(rotated[1], x_single[0], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 8, 16])
def test_various_batch_sizes(batch_size):
    """Ensure consistency across different batch sizes."""
    x = torch.randn(batch_size, 3, 50)
    ux, uy, uz = torch.zeros(batch_size), torch.zeros(batch_size), torch.ones(batch_size)
    angles = torch.rand(batch_size) * np.pi

    rotated = rotate_batch_3d(x, ux, uy, uz, angles)
    assert rotated.shape == (batch_size, 3, 50)

    # Each time step should have preserved norm
    norm_in = torch.norm(x, dim=1)
    norm_out = torch.norm(rotated, dim=1)
    torch.testing.assert_close(norm_in, norm_out, atol=1e-5, rtol=1e-5)


def test_add_gaussian_noise():
    """Verify that Gaussian noise is only added during training."""
    from ml.src.augmentation import AddGaussianNoise

    x = torch.randn(4, 3, 50)

    # 1. Eval mode -> identity
    layer = AddGaussianNoise(std=0.1)
    layer.eval()
    out_eval = layer(x)
    torch.testing.assert_close(x, out_eval)

    # 2. Train mode -> noise added
    layer.train()
    out_train = layer(x)
    assert not torch.allclose(x, out_train)
    assert out_train.shape == x.shape


def test_random_scaling():
    """Verify scaling logic and shape consistency in train/eval."""
    from ml.src.augmentation import RandomScaling

    # Use positive values far from zero to ensure stable division for test assertions
    x = torch.randn(4, 3, 50).abs() + 0.5

    layer = RandomScaling(min_scale=0.8, max_scale=1.2)

    # Eval mode
    layer.eval()
    torch.testing.assert_close(x, layer(x))

    # Train mode
    layer.train()
    out = layer(x)
    assert out.shape == x.shape

    # Extract scale factor using the first element of each batch
    scale_factor = out[:, 0:1, 0:1] / x[:, 0:1, 0:1]  # shape (B, 1, 1)

    # The entire output should be exactly scaled by this factor
    torch.testing.assert_close(out, x * scale_factor, atol=1e-5, rtol=1e-5)


def test_bias_shift():
    """Verify bias shift logic and shape consistency in train/eval."""
    from ml.src.augmentation import BiasShift

    x = torch.randn(4, 3, 50)

    layer = BiasShift(max_shift=0.2)

    # Eval mode
    layer.eval()
    torch.testing.assert_close(x, layer(x))

    # Train mode
    layer.train()
    out = layer(x)
    assert out.shape == x.shape

    # Check that within a sample and channel, the shift is constant across time
    diff = out - x
    # Std across the sequence length (dimension 2) should be 0 (constant shift)
    std_diff = torch.std(diff, dim=2)
    torch.testing.assert_close(std_diff, torch.zeros_like(std_diff), atol=1e-6, rtol=1e-6)


def test_augmentation_config_parsing():
    """Verify that Hydra correctly parses the default augmentation configuration."""
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="config")

        # Verify default augmentations structure exists
        assert "augmentations" in cfg.data
        assert cfg.data.augmentations.rotation is True
        assert cfg.data.augmentations.noise is False
        assert cfg.data.augmentations.scaling is False
        assert cfg.data.augmentations.bias is False

        # Verify default parameters
        assert cfg.data.augment_params.noise_std == 0.02
        assert cfg.data.augment_params.min_scale == 0.8
        assert cfg.data.augment_params.max_scale == 1.2
        assert cfg.data.augment_params.max_shift == 0.1
