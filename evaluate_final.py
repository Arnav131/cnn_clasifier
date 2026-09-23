#!/usr/bin/env python3
"""Final, one-time evaluation on the untouched final_test split (15%).

Run this exactly once, after run_final.py has produced best.pt. It produces:
  - results/accuracy_report.md   (deliverable #8)
  - results/accuracy_report.json (deliverable #9)
  - results/confusion_matrix.png (deliverable #10)
  - results/error_analysis.md    (deliverable #15)

Usage:
    python evaluate_final.py --dataset-dir dataset --output-dir .
"""
import argparse
import os

from pipeline import config
from pipeline.data import build_manifest, load_classes
from pipeline.evaluate import (
    compute_metrics,
    error_analysis,
    load_model_from_checkpoint,
    plot_confusion_matrix,
    run_inference,
    write_accuracy_report,
)
from pipeline.data import get_dataloader
from pipeline.utils import get_device, leakage_report, set_seed


def main():
    parser = argparse.ArgumentParser(description="One-time final_test evaluation of best.pt.")
    parser.add_argument("--dataset-dir", default=config.DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=None,
                         help=f"Defaults to checkpoints/{config.FINAL_CHECKPOINT_NAME}")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    splits_dir = os.path.join(args.output_dir, config.DEFAULT_SPLITS_DIR)
    checkpoint_dir = os.path.join(args.output_dir, config.DEFAULT_CHECKPOINT_DIR)
    results_dir = os.path.join(args.output_dir, config.DEFAULT_RESULTS_DIR)
    config.ensure_dirs(results_dir)

    checkpoint_path = args.checkpoint or os.path.join(checkpoint_dir, config.FINAL_CHECKPOINT_NAME)

    df = build_manifest(args.dataset_dir, splits_dir, seed=args.seed)
    print(leakage_report(df.to_dict("records")))
    classes = load_classes(splits_dir)

    device = get_device()
    print(f"Device: {device}")

    model, ckpt = load_model_from_checkpoint(checkpoint_path, num_classes=len(classes), device=device)
    print(f"Loaded checkpoint from {checkpoint_path} (trained to epoch {ckpt.get('epoch')}, "
          f"dev_val acc at save time = {ckpt.get('val_acc'):.4f})")

    test_loader = get_dataloader(
        df, args.dataset_dir, "final_test", args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    n_test = sum(df.split == "final_test")
    print(f"\n=== Evaluating ONCE on final_test ({n_test} images, never used before now) ===")

    y_true, y_pred, filepaths, probs = run_inference(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, classes)

    print(f"final_test accuracy: {metrics['accuracy']*100:.2f}%  "
          f"macro_f1: {metrics['macro_f1']*100:.2f}%")

    plot_confusion_matrix(y_true, y_pred, classes, os.path.join(results_dir, "confusion_matrix.png"))

    context = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": ckpt.get("epoch"),
        "dev_val_acc_at_save_time": round(float(ckpt.get("val_acc", 0.0)), 4),
        "hparams": ckpt.get("hparams"),
        "num_classes": len(classes),
        "final_test_size": n_test,
    }
    write_accuracy_report(
        metrics,
        os.path.join(results_dir, "accuracy_report.md"),
        os.path.join(results_dir, "accuracy_report.json"),
        context,
    )

    error_analysis(
        y_true, y_pred, classes, filepaths,
        os.path.join(results_dir, "error_analysis.md"),
    )

    print(f"\nWrote: {os.path.join(results_dir, 'accuracy_report.md')}")
    print(f"Wrote: {os.path.join(results_dir, 'accuracy_report.json')}")
    print(f"Wrote: {os.path.join(results_dir, 'confusion_matrix.png')}")
    print(f"Wrote: {os.path.join(results_dir, 'error_analysis.md')}")


if __name__ == "__main__":
    main()
