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

from ml.src.augmentation import Random3DRotation
from ml.src.model import MinMaxNormalize, TinyMLConvNet, VectorNorm


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> float:
    """
    Main training execution function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Experiment: Architecture {cfg.model.filters} | Mode Norm: {cfg.data.use_norm}")

    base_model = []
    augmentation = []

    if cfg.data.augment:
        # Add random 3D rotation to the base model
        augmentation.append(Random3DRotation())

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
    print(f"📐 Total Model Parameters: {total_params:,}")

    # 2. Dataset Setup via Hydra (No-Split configuration logic)
    train_dataset = hydra.utils.instantiate(cfg.data.train_ds)
    val_dataset = hydra.utils.instantiate(cfg.data.val_ds)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # 3. Training Loop
    writer = SummaryWriter(log_dir=".")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.parameters(), lr=cfg.training.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    final_acc = 0.0
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 15
    for epoch in range(cfg.training.epochs):
        # TRAIN MODE
        train_pipeline.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{cfg.training.epochs}", leave=False)
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

        with torch.no_grad():
            for v_inputs, v_targets in val_loader:
                v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                v_outputs = base_model(v_inputs)

                loss = criterion(v_outputs, v_targets)
                val_loss += loss.item()

                preds = torch.argmax(v_outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(v_targets.cpu().numpy())

                correct += (preds == v_targets).sum().item()
                total += v_targets.size(0)

        final_acc = correct / total if total > 0 else 0
        final_f1 = f1_score(all_targets, all_preds, average="macro")
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:2d} | "
            f"Train Loss: {epoch_loss / len(train_loader):.5f} | "
            f"Val Loss: {avg_val_loss:.5f} | "
            f"Val Acc: {final_acc:.2%} | "
            f"F1: {final_f1:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        writer.add_scalar("Accuracy/val", final_acc, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("F1_Score/val", final_f1, epoch)
        writer.add_scalar("LR/train", current_lr, epoch)

        # LR Scheduler Step
        scheduler.step(final_acc)

        # Early Stopping
        if final_acc > best_acc:
            best_acc = final_acc
            patience_counter = 0
            # Save the best model locally
            os.makedirs("models", exist_ok=True)
            suffix = "_norm" if cfg.data.use_norm else "_3d"
            torch.save(base_model.state_dict(), f"models/best_pipeline{suffix}.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
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
    # Return best_acc so Optuna solves to maximize the true peak performance
    return float(best_acc)


if __name__ == "__main__":
    train()
