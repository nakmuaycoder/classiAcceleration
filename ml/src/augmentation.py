"""
3D Rotation Augmentation for Accelerometer Data.

This module provides utilities to rotate 3D time-series data using
the Rodrigues' rotation formula.

It is designed to make TinyML models invariant to sensor orientation.

Author: nakmuaycoder
Date: 2026/04
"""

import numpy as np
import torch
import torch.nn as nn


def rotate_batch_3d(
    x: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    """
    Applies 3D rotation to a batch of accelerometer data using Rodrigues' formula.

    Args:
        x (Tensor): Input data of shape (B, 3, L) or (3, L).
        ux, uy, uz (Tensor): Components of the rotation axis (B,).
        angles (Tensor): Rotation angles in radians (B,).

    Returns:
        Tensor: Rotated data of same shape as input.
    """
    # Ensure 3D (B, 3, L)
    original_shape = x.shape
    if x.dim() == 2:
        x = x.unsqueeze(0)

    # Rodrigues' rotation formula components
    c = torch.cos(angles)
    s = torch.sin(angles)
    t = 1 - c

    # Construct 3x3 Rotation matrices (B, 3, 3)
    r00 = t * ux * ux + c
    r01 = t * ux * uy - s * uz
    r02 = t * ux * uz + s * uy

    r10 = t * ux * uy + s * uz
    r11 = t * uy * uy + c
    r12 = t * uy * uz - s * ux

    r20 = t * ux * uz - s * uy
    r21 = t * uy * uz + s * ux
    r22 = t * uz * uz + c

    rot_matrix = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=1),
            torch.stack([r10, r11, r12], dim=1),
            torch.stack([r20, r21, r22], dim=1),
        ],
        dim=1,
    )

    # Apply rotation: (B, 3, 3) @ (B, 3, L) -> (B, 3, L)
    rotated_x = torch.bmm(rot_matrix, x)

    return rotated_x.view(original_shape)


class Random3DRotation(nn.Module):
    """
    PyTorch Augmentation Layer that applies random 3D rotations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Samples random rotation parameters and applies them to the batch.
        """
        if not self.training:
            return x

        # Ensure batch info
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        device = x.device

        # 1. Sample random unit rotation axis (Uniform on sphere)
        phi = torch.rand(batch_size, device=device) * 2 * np.pi
        costheta = torch.rand(batch_size, device=device) * 2 - 1
        theta = torch.acos(costheta)

        ux = torch.sin(theta) * torch.cos(phi)
        uy = torch.sin(theta) * torch.sin(phi)
        uz = torch.cos(theta)

        # 2. Sample random rotation angle [0, 2pi]
        angles = torch.rand(batch_size, device=device) * 2 * np.pi

        # 3. Apply rotation
        return rotate_batch_3d(x, ux, uy, uz, angles)
