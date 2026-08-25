"""
VoiceShield Neural Step 3: End-to-End Deep Learning Training & Fine-Tuning Engine.
Features:
  - Binary Focal Loss with Logits
  - Differential Parameter Groups (Low LR for Pretrained Backbone, High LR for Attention Head)
  - Cosine Annealing Learning Rate Schedule with Warmup
  - Mixed-Precision Hardware Acceleration (AMP & GradScaler)
  - Full Validation Loop Tracking EER, ROC-AUC, Macro-F1, and Confusion Matrices
  - Early Stopping and Best Checkpoint Persistence
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import MANIFEST_PATH
from src.dataset_loader import create_dataloaders
from src.losses import BinaryFocalLossWithLogits
from src.neural_model import VoiceShieldNeuralDetector

DEFAULT_OUTPUT_MODEL: str = "models/voiceshield_neural_best.pt"
DEFAULT_OUTPUT_META: str = "models/neural_metadata.json"


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and operating threshold where FPR == FNR.
    """
    if len(set(y_true)) < 2:
        return 0.0, 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, eer_thresh


def evaluate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Runs full evaluation over validation dataset split.
    """
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for audio, targets, _ in val_loader:
            audio = audio.to(device)
            targets = targets.to(device)

            logits = model(audio)
            loss = criterion(logits, targets)
            val_loss += loss.item() * audio.size(0)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(targets_np)

    total_samples = max(1, len(all_targets))
    mean_loss = float(val_loss / total_samples)

    y_true = np.array(all_targets, dtype=int)
    y_probs = np.array(all_probs, dtype=float)

    eer, eer_thresh = compute_eer(y_true, y_probs)
    opt_thresh = float(np.clip(eer_thresh, 0.20, 0.80)) if not np.isnan(eer_thresh) else 0.50

    y_pred = (y_probs >= opt_thresh).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) > 1 else acc
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    auc = float(roc_auc_score(y_true, y_probs)) if len(set(y_true)) > 1 else 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "loss": mean_loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "eer": eer,
        "optimal_threshold": opt_thresh,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_neural_model(
    manifest_path: str = MANIFEST_PATH,
    backbone: str = "lightweight",
    epochs: int = 15,
    batch_size: int = 8,
    lr_backbone: float = 1e-5,
    lr_head: float = 1e-4,
    weight_decay_backbone: float = 0.01,
    weight_decay_head: float = 0.001,
    val_split: float = 0.20,
    patience: int = 4,
    output_model_path: str = DEFAULT_OUTPUT_MODEL,
    output_metadata_path: str = DEFAULT_OUTPUT_META,
    device_name: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes full neural training loop with early stopping, mixed precision, and checkpointing.
    """
    # 1. Deterministic Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Hardware Device & AMP Setup
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    print("\n" + "=" * 70)
    print("      VOICESHIELD NEURAL TRANSFORMER TRAINING ENGINE (STEP 3)")
    print("=" * 70)
    print(f" • Device               : {device} (AMP Enabled: {use_cuda})")
    print(f" • Backbone Architecture: {backbone}")
    print(f" • Dataset Manifest     : {manifest_path}")
    print(f" • Epochs               : {epochs} (Early Stopping Patience: {patience})")
    print(f" • Batch Size           : {batch_size}")
    print(f" • Learning Rates       : Backbone: {lr_backbone:.1e} | Head: {lr_head:.1e}")

    # 3. Create PyTorch Dataloaders
    train_loader, val_loader = create_dataloaders(
        manifest_path=manifest_path,
        batch_size=batch_size,
        num_workers=0,
        val_split=val_split,
        seed=seed,
    )
    print(f" • Train Samples        : {len(train_loader.dataset)} across {len(train_loader)} batches")
    print(f" • Validation Samples   : {len(val_loader.dataset)} across {len(val_loader)} batches")

    # 4. Instantiate Model & Loss
    model = VoiceShieldNeuralDetector(backbone_name=backbone, device=device)
    criterion = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25)

    # 5. Differential Parameter Groups
    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone, "weight_decay": weight_decay_backbone},
            {
                "params": list(model.attn_pool.parameters()) + list(model.head.parameters()),
                "lr": lr_head,
                "weight_decay": weight_decay_head,
            },
        ]
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 6. Training Loop with Metric Tracking
    best_val_eer = float("inf")
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0
    history = []

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (audio, targets, _) in enumerate(train_loader):
            audio = audio.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if use_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(audio)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(audio)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * audio.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        scheduler.step()

        # Validation Step
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        epoch_dur = time.time() - epoch_start

        current_val_eer = val_metrics["eer"]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_metrics["loss"], 5),
            "val_accuracy": round(val_metrics["accuracy"] * 100, 2),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "val_roc_auc": round(val_metrics["roc_auc"], 4),
            "val_eer": round(current_val_eer * 100, 2),
            "duration_sec": round(epoch_dur, 2),
        })

        print(
            f" [Epoch {epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']*100:.1f}% | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val EER: {current_val_eer*100:.2f}% | "
            f"({epoch_dur:.1f}s)"
        )

        # Checkpoint if Validation EER Improved
        if current_val_eer < best_val_eer or (abs(current_val_eer - best_val_eer) < 1e-4 and val_metrics["macro_f1"] > best_metrics.get("macro_f1", 0.0)):
            best_val_eer = current_val_eer
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0

            # Save checkpoint
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "architecture": "VoiceShieldNeuralDetector (Wav2Vec2/Conv1D + MultiHeadAttn + DenseHead)",
                },
                output_model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch} (No EER improvement for {patience} epochs).")
                break

    total_time = time.time() - start_train_time
    print("\n" + "=" * 70)
    print("               NEURAL MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f" • Best Checkpoint Epoch : {best_epoch}")
    print(f" • Best Validation EER   : {best_val_eer * 100:.2f}%")
    print(f" • Best Macro F1-Score   : {best_metrics.get('macro_f1', 0.0):.4f}")
    print(f" • Best Validation AUC   : {best_metrics.get('roc_auc', 0.0):.4f}")
    print(f" • Total Training Time   : {total_time:.2f} seconds")

    # 7. Persist Metadata JSON
    model_sha256 = hashlib.sha256(open(output_model_path, "rb").read()).hexdigest() if os.path.exists(output_model_path) else ""
    metadata = {
        "model_name": "VoiceShield Deep Learning Neural Classifier",
        "model_version": "3.0.0",
        "architecture": "VoiceShieldNeuralDetector (MultiHeadAttn + BinaryLogitHead)",
        "model_artifact_path": output_model_path,
        "model_artifact_sha256": model_sha256,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_epochs_trained": epoch,
        "best_epoch": best_epoch,
        "training_duration_seconds": round(total_time, 2),
        "best_metrics": best_metrics,
        "training_history": history,
        "hyperparameters": {
            "loss": "BinaryFocalLossWithLogits (gamma=2.0, alpha=0.25)",
            "batch_size": batch_size,
            "lr_backbone": lr_backbone,
            "lr_head": lr_head,
            "weight_decay_backbone": weight_decay_backbone,
            "weight_decay_head": weight_decay_head,
            "scheduler": "CosineAnnealingLR",
        },
    }

    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Best model saved to: {output_model_path}")
    print(f"[OK] Metadata saved to  : {output_metadata_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Neural Deep Learning Training Engine")
    parser.add_argument("--manifest", type=str, default=MANIFEST_PATH, help="Path to manifest.csv")
    parser.add_argument("--backbone", type=str, default="lightweight", choices=["lightweight", "facebook/wav2vec2-base", "microsoft/wavlm-base"], help="Speech backbone")
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Backbone learning rate")
    parser.add_argument("--lr-head", type=float, default=1e-4, help="Classifier head learning rate")
    parser.add_argument("--val-split", type=float, default=0.20, help="Validation partition proportion")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--output-model", type=str, default=DEFAULT_OUTPUT_MODEL, help="Target weights file (.pt)")
    parser.add_argument("--output-metadata", type=str, default=DEFAULT_OUTPUT_META, help="Target metadata file (.json)")
    parser.add_argument("--device", type=str, default=None, help="Hardware device (cuda / cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")

    args = parser.parse_args()
    train_neural_model(
        manifest_path=args.manifest,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        val_split=args.val_split,
        patience=args.patience,
        output_model_path=args.output_model,
        output_metadata_path=args.output_metadata,
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
