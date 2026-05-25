"""
TinyML Training Script with Advanced Pipeline Composition.
Unified Training Pipeline: [Augment] -> [Normalization] -> [CNN Model]

Author: nakmuaycoder
Date: 2026/04
"""

import os

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ml.src.augmentation import (
    AddGaussianNoise,
    BiasShift,
    Random3DRotation,
    RandomScaling,
)
from ml.src.model import MinMaxNormalize, TinyMLConvNet, VectorNorm


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> float:
    """
    Main training execution function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_multirun = False
    trial_num = None
    try:
        from hydra.core.hydra_config import HydraConfig

        is_multirun = HydraConfig.get().mode.name == "MULTIRUN"
        trial_num = HydraConfig.get().job.num
    except Exception:
        pass

    if is_multirun:
        print(
            f"[Trial {trial_num}] 🚀 Starting: "
            f"filters={cfg.model.filters}, "
            f"lr={cfg.training.lr:.4f}, "
            f"seq_len={cfg.data.seq_len}, "
            f"use_norm={cfg.data.use_norm}"
        )
    else:
        print(f"🚀 Experiment: Architecture {cfg.model.filters} | Mode Norm: {cfg.data.use_norm}")

    base_model = []
    augmentation = []

    if cfg.data.augment:
        # Build the active list of augmentations dynamically
        augmentations_config = cfg.data.get("augmentations", {})

        # 1. Random 3D Rotation (default to True for backwards compatibility)
        if augmentations_config.get("rotation", True):
            augmentation.append(Random3DRotation())

        # 2. Gaussian Noise
        if augmentations_config.get("noise", False):
            std = cfg.data.get("augment_params", {}).get("noise_std", 0.02)
            augmentation.append(AddGaussianNoise(std=std))

        # 3. Random Scaling
        if augmentations_config.get("scaling", False):
            min_scale = cfg.data.get("augment_params", {}).get("min_scale", 0.8)
            max_scale = cfg.data.get("augment_params", {}).get("max_scale", 1.2)
            augmentation.append(RandomScaling(min_scale=min_scale, max_scale=max_scale))

        # 4. Bias Shift
        if augmentations_config.get("bias", False):
            max_shift = cfg.data.get("augment_params", {}).get("max_shift", 0.1)
            augmentation.append(BiasShift(max_shift=max_shift))

    # 1. Model & Pipeline Definition
    if cfg.data.use_norm:
        in_channels = 1
        base_model.append(VectorNorm())
    else:
        in_channels = 3

    # Add min-max normalization to the base model
    base_model.append(MinMaxNormalize())

    base_model.append(
        TinyMLConvNet(
            in_channels=in_channels,
            filters=cfg.model.filters,
            num_classes=3,
            seq_len=cfg.data.seq_len,
            hidden_dim=cfg.model.hidden_dim,
        )
    )

    base_model = nn.Sequential(*base_model).to(device)
    train_pipeline = nn.Sequential(*augmentation, base_model).to(device)

    total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    if not is_multirun:
        print(f"📐 Total Model Parameters: {total_params:,}")

    # 2. Dataset Setup via Hydra (No-Split configuration logic)
    train_dataset = hydra.utils.instantiate(cfg.data.train_ds)
    val_dataset = hydra.utils.instantiate(cfg.data.val_ds)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # 3. Training Loop
    writer = SummaryWriter(log_dir=".")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(train_pipeline.parameters(), lr=cfg.training.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    final_acc = 0.0
    best_acc = 0.0
    best_acc_rot = 0.0
    patience_counter = 0
    early_stop_patience = 15
    val_rotator = Random3DRotation()
    val_rotator.train()  # Force rotation logic to run
    for epoch in range(cfg.training.epochs):
        # TRAIN MODE
        train_pipeline.train()
        epoch_loss = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:2d}/{cfg.training.epochs}",
            leave=False,
            disable=is_multirun,
        )
        for raw_inputs, targets in pbar:
            raw_inputs, targets = raw_inputs.to(device), targets.to(device)

            # Simple call to the ONE training pipeline
            outputs = train_pipeline(raw_inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # EVAL MODE (No data augmentation applied)
        base_model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        all_preds = []
        all_targets = []

        # ROTATED EVAL MODE
        correct_rot, total_rot = 0, 0
        val_loss_rot = 0.0
        all_preds_rot = []
        all_targets_rot = []

        with torch.no_grad():
            for v_inputs, v_targets in val_loader:
                v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)

                # Standard Evaluation
                v_outputs = base_model(v_inputs)
                loss = criterion(v_outputs, v_targets)
                val_loss += loss.item()

                preds = torch.argmax(v_outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(v_targets.cpu().numpy())

                correct += (preds == v_targets).sum().item()
                total += v_targets.size(0)

                # Rotated Evaluation
                v_inputs_rot = val_rotator(v_inputs)
                v_outputs_rot = base_model(v_inputs_rot)
                loss_rot = criterion(v_outputs_rot, v_targets)
                val_loss_rot += loss_rot.item()

                preds_rot = torch.argmax(v_outputs_rot, dim=1)
                all_preds_rot.extend(preds_rot.cpu().numpy())
                all_targets_rot.extend(v_targets.cpu().numpy())

                correct_rot += (preds_rot == v_targets).sum().item()
                total_rot += v_targets.size(0)

        final_acc = correct / total if total > 0 else 0
        final_f1 = f1_score(all_targets, all_preds, average="macro")
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        final_acc_rot = correct_rot / total_rot if total_rot > 0 else 0
        final_f1_rot = f1_score(all_targets_rot, all_preds_rot, average="macro")
        avg_val_loss_rot = val_loss_rot / len(val_loader) if len(val_loader) > 0 else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        if not is_multirun:
            print(
                f"Epoch {epoch:2d} | Train Loss: {epoch_loss / len(train_loader):.4f} | "
                f"Val Acc: {final_acc:.2%} (Rot: {final_acc_rot:.2%}) | "
                f"Val F1: {final_f1:.4f} (Rot: {final_f1_rot:.4f}) | "
                f"LR: {current_lr:.2e}"
            )

        writer.add_scalar("Accuracy/val", final_acc, epoch)
        writer.add_scalar("Accuracy/val_rotated", final_acc_rot, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Loss/val_rotated", avg_val_loss_rot, epoch)
        writer.add_scalar("F1_Score/val", final_f1, epoch)
        writer.add_scalar("F1_Score/val_rotated", final_f1_rot, epoch)
        writer.add_scalar("LR/train", current_lr, epoch)

        # LR Scheduler Step
        scheduler.step(final_acc)

        # Early Stopping
        if final_acc > best_acc:
            best_acc = final_acc
            best_acc_rot = final_acc_rot
            patience_counter = 0
            # Save the best model locally in the run directory
            os.makedirs("models", exist_ok=True)
            suffix = "_norm" if cfg.data.use_norm else "_3d"
            torch.save(base_model.state_dict(), f"models/best_pipeline{suffix}.pth")

            # Check and save the best model overall in output/best
            best_dir = os.path.join(cfg.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            best_acc_file = os.path.join(best_dir, "best_accuracy.txt")

            # Read overall best accuracy across runs/trials
            overall_best_acc = 0.0
            if os.path.exists(best_acc_file):
                try:
                    with open(best_acc_file) as f:
                        overall_best_acc = float(f.read().strip())
                except Exception:
                    pass

            if final_acc > overall_best_acc:
                overall_best_acc = final_acc
                best_model_path = os.path.join(best_dir, f"best_pipeline{suffix}.pth")
                torch.save(base_model.state_dict(), best_model_path)
                try:
                    with open(best_acc_file, "w") as f:
                        f.write(f"{overall_best_acc:.6f}\n")
                except Exception as e:
                    print(f"Warning: Could not write overall best accuracy: {e}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if not is_multirun:
                    print(
                        f"🛑 Early stopping triggered after {epoch} epochs "
                        f"(Patience {early_stop_patience})."
                    )
                break

    # 4. Save Final Weights & Export TensorBoard HParams
    os.makedirs("models", exist_ok=True)
    suffix = "_norm" if cfg.data.use_norm else "_3d"
    torch.save(base_model.state_dict(), f"models/trained_pipeline{suffix}_last.pth")

    hparams = {
        "lr": cfg.training.lr,
        "batch_size": cfg.training.batch_size,
        "filters_0": cfg.model.filters[0],
        "filters_1": cfg.model.filters[1],
        "hidden_dim": cfg.model.hidden_dim,
        "augment": cfg.data.augment,
        "use_norm": cfg.data.use_norm,
        "total_params": total_params,
    }
    writer.add_hparams(hparams, {"hparam/accuracy": best_acc, "hparam/f1": final_f1})

    writer.close()
    if is_multirun:
        print(
            f"[Trial {trial_num}] 🎉 Finished | "
            f"Best Val Acc: {best_acc:.2%} (Rot: {best_acc_rot:.2%}) | "
            f"Params: {total_params:,}"
        )
    # Return best_acc so Optuna solves to maximize the true peak performance
    return float(best_acc)


if __name__ == "__main__":
    train()
