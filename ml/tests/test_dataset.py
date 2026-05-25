"""
Unit Tests for the AccelerometerDataset.

Tests include:
1. File loading and windowing calculation.
2. Tensor shape & channel first convention.
3. Label alignment.

Author: nakmuaycoder
Date: 2026/04
"""

import numpy as np
import pandas as pd
import pytest
import torch

from ml.src.dataset import AccelerometerDataset


@pytest.fixture
def dummy_csv(tmp_path):
    """Creates a dummy CSV file for dataset testing."""
    df = pd.DataFrame(
        {
            "x": np.arange(100, dtype=np.float32),
            "y": np.arange(100, dtype=np.float32) * 10,
            "z": np.arange(100, dtype=np.float32) * 100,
            "label": (np.arange(100) > 50).astype(int),  # 0 for first 50, 1 for rest
        }
    )
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_dataset_windowing(dummy_csv):
    """Verify windowing count: (Length - seq_len) / stride."""
    seq_len = 10
    stride = 2
    # (100 - 10) / 2 = 45 windows

    ds = AccelerometerDataset(files=[dummy_csv], seq_len=seq_len, stride=stride)

    assert len(ds) == 45
    print(f"✅ Windowing count check: {len(ds)} == 45.")


def test_getitem_shapes(dummy_csv):
    """Check Tensor shape: (3, seq_len)."""
    seq_len = 10
    ds = AccelerometerDataset(files=[dummy_csv], seq_len=seq_len)

    x, y = ds[0]

    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, seq_len)
    assert isinstance(y, torch.Tensor)
    print("✅ Getitem shape check passed.")


def test_data_content(dummy_csv):
    """Check if the data and label values are correctly aligned."""
    seq_len = 10
    stride = 2
    ds = AccelerometerDataset(files=[dummy_csv], seq_len=seq_len, stride=stride)

    # First window index 0 starts at point 0
    x, y = ds[0]
    # x[0] should be range(0, 10)
    torch.testing.assert_close(x[0], torch.arange(10, dtype=torch.float32))
    # Label should be point 5 (label 0)
    assert y.item() == 0

    # Window 30 starts at index 60 (30 * 2)
    x_late, y_late = ds[30]
    # Label at 60 + 5 = 65 should be 1
    assert y_late.item() == 1
    print("✅ Data alignment and label check passed.")


def test_empty_dataset():
    """Ensure no crash when given an empty list of files."""
    ds = AccelerometerDataset(files=[], seq_len=10)
    assert len(ds) == 0
    print("✅ Empty dataset check passed.")


def test_dataset_pca(dummy_csv):
    """Verify that PCA alignment preserves shapes and matches the sklearn pipeline."""
    seq_len = 10

    # Load dataset with PCA
    ds_pca = AccelerometerDataset(files=[dummy_csv], seq_len=seq_len, use_pca=True)
    # Load dataset without PCA (normal)
    ds_norm = AccelerometerDataset(files=[dummy_csv], seq_len=seq_len, use_pca=False)

    assert len(ds_pca) == len(ds_norm)

    x_pca, y_pca = ds_pca[0]
    x_norm, y_norm = ds_norm[0]

    # Shape check
    assert x_pca.shape == (3, seq_len)
    assert y_pca == y_norm

    # Check that PCA components are orthogonal and centered
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(dummy_csv)
    raw_data = df[["x", "y", "z"]].values.astype(np.float32)
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=3))
    expected_full = pipeline.fit_transform(raw_data).astype(np.float32)
    expected_window = expected_full[:seq_len].T

    torch.testing.assert_close(x_pca, torch.from_numpy(expected_window))
    print("✅ PCA dataset transformation check passed.")
