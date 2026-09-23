"""Evaluation utilities: metrics, confusion matrix, training curves, MOGEO
convergence plot, accuracy report (md + json), and error analysis."""
from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .model import build_model


@torch.no_grad()
def run_inference(model, loader, device) -> Tuple[List[int], List[int], List[str], np.ndarray]:
    model.eval()
    y_true, y_pred, filepaths = [], [], []
    all_probs = []
    for images, labels, paths in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        filepaths.extend(list(paths))
        all_probs.append(probs.cpu().numpy())
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 0))
    return y_true, y_pred, filepaths, probs_arr


def load_model_from_checkpoint(checkpoint_path: str, num_classes: int, device) -> Tuple[torch.nn.Module, Dict]:
    from .config import HParams
    from .utils import load_checkpoint

    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    hparams = HParams.from_dict(ckpt["hparams"])
    model = build_model(hparams, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def compute_metrics(y_true: List[int], y_pred: List[int], classes: List[str]) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    top1_correct = int(np.sum(np.array(y_true) == np.array(y_pred)))
    report_dict = classification_report(
        y_true, y_pred, target_names=classes, zero_division=0, output_dict=True
    )
    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_samples": len(y_true),
        "num_correct": top1_correct,
        "per_class_report": report_dict,
    }


def plot_confusion_matrix(y_true, y_pred, classes: List[str], out_path: str, max_labels_shown: int = 40) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    n = len(classes)
    show_labels = n <= max_labels_shown
    fig_size = max(10, min(30, n * 0.28))
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix ({n} classes, final_test set)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar(fraction=0.03)
    if show_labels:
        plt.xticks(range(n), classes, rotation=90, fontsize=6)
        plt.yticks(range(n), classes, fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], out_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train_acc")
    axes[1].plot(epochs, history["val_acc"], label="val_acc")
    if "val_macro_f1" in history:
        axes[1].plot(epochs, history["val_macro_f1"], label="val_macro_f1", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Accuracy / F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mogeo_convergence(mogeo_results_df: pd.DataFrame, out_path: str) -> None:
    """Plots best-so-far val_acc and val_macro_f1 per generation, plus the
    number of non-dominated (Pareto-optimal) candidates found each generation."""
    grouped = mogeo_results_df.groupby("generation").agg(
        best_val_acc=("val_acc", "max"),
        best_val_macro_f1=("val_macro_f1", "max"),
    ).reset_index()
    grouped["best_val_acc_so_far"] = grouped["best_val_acc"].cummax()
    grouped["best_val_macro_f1_so_far"] = grouped["best_val_macro_f1"].cummax()

    plt.figure(figsize=(9, 6))
    plt.plot(grouped["generation"], grouped["best_val_acc_so_far"], marker="o", label="best val_acc so far")
    plt.plot(grouped["generation"], grouped["best_val_macro_f1_so_far"], marker="s", label="best val_macro_f1 so far")
    plt.scatter(mogeo_results_df["generation"], mogeo_results_df["val_acc"], alpha=0.3, s=15, label="candidate val_acc")
    plt.xlabel("MOGEO generation")
    plt.ylabel("Score")
    plt.title("MOGEO Convergence (dev_val set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_accuracy_report(
    metrics: Dict,
    out_md_path: str,
    out_json_path: str,
    context: Dict,
) -> None:
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "context": context}, f, indent=2)

    lines = []
    lines.append("# Accuracy Report\n")
    lines.append(f"**Evaluated on:** `final_test` split ({metrics['num_samples']} images, "
                 f"never used during MOGEO optimization or training)\n")
    lines.append("## Summary\n")
    lines.append(f"- **Top-1 Accuracy:** {metrics['accuracy']*100:.2f}% "
                 f"({metrics['num_correct']}/{metrics['num_samples']})")
    lines.append(f"- **Macro Precision:** {metrics['macro_precision']*100:.2f}%")
    lines.append(f"- **Macro Recall:** {metrics['macro_recall']*100:.2f}%")
    lines.append(f"- **Macro F1:** {metrics['macro_f1']*100:.2f}%")
    lines.append(f"- **Weighted F1:** {metrics['weighted_f1']*100:.2f}%\n")

    lines.append("## Context\n")
    for k, v in context.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")

    lines.append("## Notes on validity\n")
    lines.append(
        "This report is generated exclusively from real model inference on the "
        "untouched 15% final_test split, computed once, after MOGEO hyperparameter "
        "search and final training were both completed. No numbers in this file are "
        "simulated, hardcoded, or copied from literature. Compare this to the legacy "
        "`MainCode.ipynb` \"Performance\" section, which generated accuracy/precision/"
        "recall/F1 via `np.random.uniform(0.98, 0.9989)` and then sorted the random "
        "values to fake an improving trend -- that section was fabricated and is "
        "superseded by this report.\n"
    )

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def error_analysis(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    filepaths: List[str],
    out_path: str,
    top_k_confusions: int = 20,
    max_examples_per_class: int = 5,
) -> None:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Most confused class pairs (true -> predicted, excluding correct)
    confusion_pairs = Counter()
    misclassified_examples: Dict[Tuple[int, int], List[str]] = {}
    for t, p, fp in zip(y_true, y_pred, filepaths):
        if t != p:
            confusion_pairs[(t, p)] += 1
            misclassified_examples.setdefault((t, p), []).append(fp)

    # Per-class accuracy, to surface worst-performing classes
    per_class_acc = {}
    for c in range(len(classes)):
        mask = y_true_arr == c
        n = mask.sum()
        if n == 0:
            continue
        per_class_acc[c] = float((y_pred_arr[mask] == c).mean())

    worst_classes = sorted(per_class_acc.items(), key=lambda x: x[1])[:20]

    lines = ["# Error Analysis\n"]
    lines.append(f"Total final_test samples: {len(y_true)}")
    lines.append(f"Total misclassified: {int(np.sum(y_true_arr != y_pred_arr))}\n")

    lines.append("## Worst-performing classes (lowest per-class recall)\n")
    lines.append("| Rank | Class | Recall | # Test Images |")
    lines.append("|---|---|---|---|")
    for rank, (c, acc) in enumerate(worst_classes, start=1):
        n = int((y_true_arr == c).sum())
        lines.append(f"| {rank} | {classes[c]} | {acc*100:.1f}% | {n} |")
    lines.append("")

    lines.append(f"## Top {top_k_confusions} most common misclassifications (true -> predicted)\n")
    lines.append("| Rank | True Class | Predicted Class | Count | Example Files |")
    lines.append("|---|---|---|---|---|")
    for rank, ((t, p), count) in enumerate(confusion_pairs.most_common(top_k_confusions), start=1):
        examples = misclassified_examples[(t, p)][:max_examples_per_class]
        examples_str = "; ".join(examples)
        lines.append(f"| {rank} | {classes[t]} | {classes[p]} | {count} | {examples_str} |")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
