"""
PyTorch Dataset for 3D Accelerometer Time-Series.

This module handles multi-file loading and sliding window sequence generation.

Author: nakmuaycoder
Date: 2026/04
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AccelerometerDataset(Dataset):
    """
    Dataset that loads specific CSV acceleration files and slices them into sequences.

    Attributes:
        seq_len (int): Window size (number of points per sample).
        stride (int): Step between windows.
    """

    def __init__(self, files: list[str], seq_len: int = 10, stride: int = 2):
        """
        Args:
            files: List of paths to the clean CSV files to load.
            seq_len: Sequence window size.
            stride: Step between samples.
        """
        self.seq_len = seq_len
        self.stride = stride
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
