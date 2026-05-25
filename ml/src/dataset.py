"""
PyTorch Dataset for 3D Accelerometer Time-Series.

This module handles multi-file loading and sliding window sequence generation.

Author: nakmuaycoder
Date: 2026/04
"""

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class AccelerometerDataset(Dataset):
    """
    Dataset that loads specific CSV acceleration files and slices them into sequences.

    Attributes:
        seq_len (int): Window size (number of points per sample).
        stride (int): Step between windows.
        use_pca (bool): Whether to perform PCA alignment per file.
    """

    def __init__(self, files: list[str], seq_len: int = 10, stride: int = 2, use_pca: bool = False):
        """
        Args:
            files: List of paths to the clean CSV files to load.
            seq_len: Sequence window size.
            stride: Step between samples.
            use_pca: If True, projects coordinates on PCA axes per file.
        """
        self.seq_len = seq_len
        self.stride = stride
        self.use_pca = use_pca
        self.samples = []
        self.labels = []

        self._load_and_window(files)

    def _load_and_window(self, files: list[str]) -> None:
        """
        Loads the provided CSV files and creates sliding windows.
        """
        if not files:
            return

        for f in files:
            df = pd.read_csv(f)
            data = df[["x", "y", "z"]].values.astype(np.float32)
            labels = df["label"].values.astype(np.int64)

            if self.use_pca:
                pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=3))
                data = pca_pipeline.fit_transform(data).astype(np.float32)

            # Sliding window allocation
            for i in range(0, len(data) - self.seq_len, self.stride):
                # Shape: (3, seq_len)
                window = data[i : i + self.seq_len].T
                label = labels[i + self.seq_len // 2]
                self.samples.append(window)
                self.labels.append(label)

        # Print summary for logging
        basename = "files" if len(files) > 1 else "file"
        print(f"✅ Loaded {len(self.samples)} sequences from {len(files)} {basename}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single sample: (channels, seq_len) and its label.
        """
        x = torch.from_numpy(self.samples[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
