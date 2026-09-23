"""AdamW training loop shared by baseline, MOGEO candidates, and final training.

Resumability: every epoch, a "last" checkpoint (model/optimizer/scheduler/
early-stopper state + history) is written to `last_checkpoint_path`. If that
file already exists when `train_one_run` starts and `resume=True`, training
continues from the saved epoch instead of restarting -- this is what makes
long final-training runs survive a Colab disconnect.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW

from . import config
from .data import get_dataloader
from .model import build_model
from .utils import EarlyStopper, load_checkpoint, save_checkpoint, set_seed


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, criterion) -> Tuple[float, float, float]:
    model.eval()
    total_loss, n = 0.0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()
    total_loss, n = 0.0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def train_one_run(
    hparams: config.HParams,
    df: pd.DataFrame,
    dataset_dir: str,
    device: torch.device,
    max_epochs: int,
    patience: int,
    best_checkpoint_path: Optional[str],
    last_checkpoint_path: Optional[str] = None,
    resume: bool = False,
    num_classes: int = config.NUM_CLASSES,
    image_size: int = config.IMAGE_SIZE,
    num_workers: int = 2,
    seed: int = config.SEED,
    verbose: bool = True,
    save_weights: bool = True,
) -> Dict:
    """Trains a single model with AdamW on dev_train, validating on dev_val.

    Returns a dict: {history, best_val_acc, best_val_macro_f1, best_epoch,
    epochs_trained, num_params, stopped_early}.
    """
    set_seed(seed)

    train_loader = get_dataloader(df, dataset_dir, "dev_train", hparams.batch_size, image_size, num_workers)
    val_loader = get_dataloader(df, dataset_dir, "dev_val", hparams.batch_size, image_size, num_workers)

    model = build_model(hparams, num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=hparams.label_smoothing)
    early_stopper = EarlyStopper(patience=patience)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }
    start_epoch = 1
    best_val_acc, best_val_macro_f1, best_epoch = 0.0, 0.0, 0

    if resume and last_checkpoint_path is not None:
        ckpt = load_checkpoint(last_checkpoint_path, map_location=device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            early_stopper.load_state_dict(ckpt["early_stopper_state"])
            history = ckpt["history"]
            start_epoch = ckpt["epoch"] + 1
            best_val_acc = ckpt["best_val_acc"]
            best_val_macro_f1 = ckpt["best_val_macro_f1"]
            best_epoch = ckpt["best_epoch"]
            if verbose:
                print(f"[resume] Resuming from epoch {start_epoch} "
                      f"(best_val_acc so far = {best_val_acc:.4f})")

    epochs_trained = start_epoch - 1
    stopped_early = False

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        epochs_trained = epoch

        is_best = early_stopper.step(val_acc)
        if is_best:
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            if save_weights and best_checkpoint_path is not None:
                save_checkpoint(
                    best_checkpoint_path,
                    {
                        "model_state": model.state_dict(),
                        "hparams": hparams.to_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "val_macro_f1": val_macro_f1,
                    },
                )

        if last_checkpoint_path is not None:
            save_checkpoint(
                last_checkpoint_path,
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "early_stopper_state": early_stopper.state_dict(),
                    "history": history,
                    "best_val_acc": best_val_acc,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_epoch": best_epoch,
                    "hparams": hparams.to_dict(),
                },
            )

        if verbose:
            print(
                f"  epoch {epoch:3d}/{max_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_macro_f1:.4f} | "
                f"{dt:.1f}s{'  * best' if is_best else ''}"
            )

        if early_stopper.should_stop:
            stopped_early = True
            if verbose:
                print(f"  Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "num_params": build_model(hparams, num_classes).num_parameters(),
        "stopped_early": stopped_early,
    }
