"""
Unit Tests for TinyML Model Components.

Tests include:
1. Shape Consistency: Input-Output matching.
2. VectorNorm: Calculation of L2 norm.
3. MinMaxNormalize: Scaling logic.
4. Filter Consistency: Parameterization check.

Author: nakmuaycoder
Date: 2026/04
"""

import pytest
import torch

from ml.src.model import MinMaxNormalize, TinyMLConvNet, VectorNorm


def test_vector_norm():
    """Test L2 norm calculation: (B, 3, L) -> (B, 1, L)"""
    norm_mod = VectorNorm()
    # Case: [3, 0, 0] -> should give norm 3
    x = torch.zeros(1, 3, 10)
    x[0, 0, :] = 3.0

    out = norm_mod(x)
    assert out.shape == (1, 1, 10)
    torch.testing.assert_close(out[0, 0, :], torch.ones(10) * 3.0)
    print("✅ VectorNorm test passed.")


def test_min_max_normalize():
    """Test scaling logic: Input [-4, 4] -> Output [0, 1]"""
    norm_mod = MinMaxNormalize(min_val=-4.0, max_val=4.0, range_min=0.0, range_max=1.0)
    x = torch.tensor([-4.0, 0.0, 4.0]).view(1, 3, 1)

    out = norm_mod(x)
    # Expected: [0.0, 0.5, 1.0]
    expected = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1)
    torch.testing.assert_close(out, expected)
    print("✅ MinMaxNormalize test passed.")


@pytest.mark.parametrize("filters, in_channels", [([8, 16], 3), ([4, 8], 1), ([16, 32], 3)])
def test_model_output_shape(filters, in_channels):
    """Test that the model produces correct output shape for various configs."""
    num_classes = 3
    seq_len = 10
    batch_size = 8

    model = TinyMLConvNet(
        in_channels=in_channels, filters=filters, num_classes=num_classes, seq_len=seq_len
    )
    x = torch.randn(batch_size, in_channels, seq_len)

    out = model(x)
    assert out.shape == (batch_size, num_classes)
    print(f"✅ Shape test passed for filters={filters}, in_channels={in_channels}.")


def test_model_parameter_count():
    """Verify that different filter counts result in different parameter volumes."""
    m1 = TinyMLConvNet(filters=[4, 8])
    m2 = TinyMLConvNet(filters=[8, 16])

    p1 = sum(p.numel() for p in m1.parameters())
    p2 = sum(p.numel() for p in m2.parameters())

    assert p1 < p2
    print(f"✅ Parameter count check: Tiny ({p1}) < Standard ({p2}).")


def test_gradients_flow():
    """Check that backpropagation works through the entire pipeline."""
    model = TinyMLConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 10)
    y = torch.tensor([0, 1, 2, 1])

    # Forward pass
    out = model(x)
    loss = criterion(out, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check if gradients are set for parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient not set for {name}"

    print("✅ Gradients flow check passed.")
