#!/usr/bin/env python3
"""Baseline experiment: fixed hyperparameters (no MOGEO), AdamW optimizer.

This isolates the effect of the pipeline rewrite (correct fixed splits, no
leakage, AdamW, real metrics) from the effect of MOGEO search, by training
an architecture close in spirit to the historical CNN (3 conv blocks, 32
base channels) but through the corrected pipeline.

Usage:
    python run_baseline.py --dataset-dir dataset --output-dir .
"""
import argparse
import os
import time

from pipeline import config
from pipeline.data import build_manifest
from pipeline.evaluate import plot_training_curves
from pipeline.train import train_one_run
from pipeline.utils import append_experiment_log, get_device, leakage_report, set_seed


def main():
    parser = argparse.ArgumentParser(description="Baseline CNN training (AdamW, fixed hyperparameters).")
    parser.add_argument("--dataset-dir", default=config.DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=config.BASELINE_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.BASELINE_EARLY_STOP_PATIENCE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    splits_dir = os.path.join(args.output_dir, config.DEFAULT_SPLITS_DIR)
    checkpoint_dir = os.path.join(args.output_dir, config.DEFAULT_CHECKPOINT_DIR)
    results_dir = os.path.join(args.output_dir, config.DEFAULT_RESULTS_DIR)
    config.ensure_dirs(splits_dir, checkpoint_dir, results_dir)

    df = build_manifest(args.dataset_dir, splits_dir, seed=args.seed)
    print(leakage_report(df.to_dict("records")))
    print(f"dev_train={sum(df.split=='dev_train')} dev_val={sum(df.split=='dev_val')} "
          f"final_test={sum(df.split=='final_test')}")

    device = get_device()
    print(f"Device: {device}")

    best_ckpt = os.path.join(checkpoint_dir, config.BASELINE_CHECKPOINT_NAME)
    last_ckpt = os.path.join(checkpoint_dir, "baseline_last.pt")
    log_path = os.path.join(results_dir, config.EXPERIMENT_LOG_FILENAME)

    hparams = config.BASELINE_HPARAMS
    print(f"Baseline hyperparameters: {hparams.to_dict()}")

    t0 = time.time()
    result = train_one_run(
        hparams=hparams,
        df=df,
        dataset_dir=args.dataset_dir,
        device=device,
        max_epochs=args.epochs,
        patience=args.patience,
        best_checkpoint_path=best_ckpt,
        last_checkpoint_path=last_ckpt,
        resume=args.resume,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    dt = time.time() - t0

    plot_training_curves(result["history"], os.path.join(results_dir, "baseline_training_curves.png"))

    append_experiment_log(log_path, {
        "run_id": "baseline",
        "stage": "baseline",
        "epochs_trained": result["epochs_trained"],
        "hparams": hparams.to_dict(),
        "best_val_acc": round(result["best_val_acc"], 4),
        "best_val_macro_f1": round(result["best_val_macro_f1"], 4),
        "final_val_loss": round(result["history"]["val_loss"][-1], 4),
        "notes": f"num_params={result['num_params']}, stopped_early={result['stopped_early']}, "
                 f"wall_time_s={dt:.1f}",
    })

    print(f"\nBaseline done in {dt:.1f}s | best_val_acc={result['best_val_acc']:.4f} "
          f"best_val_macro_f1={result['best_val_macro_f1']:.4f} (epoch {result['best_epoch']})")
    print(f"Checkpoint saved to: {best_ckpt}")


if __name__ == "__main__":
    main()
