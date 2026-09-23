#!/usr/bin/env python3
"""Final training: retrain with the best hyperparameters found by MOGEO
(or user-supplied ones) on dev_train/dev_val, up to a hard cap of
config.FINAL_MAX_EPOCHS epochs with early stopping (patience=7 by default).

final_test is NEVER used here -- only dev_train/dev_val. Produces best.pt
(deliverable #7) plus training curves. Run evaluate_final.py afterwards to
touch final_test exactly once.

Usage:
    python run_final.py --dataset-dir dataset --output-dir . --resume
"""
import argparse
import os
import time

from pipeline import config
from pipeline.data import build_manifest
from pipeline.evaluate import plot_training_curves
from pipeline.train import train_one_run
from pipeline.utils import append_experiment_log, get_device, leakage_report, load_json, set_seed


def main():
    parser = argparse.ArgumentParser(description="Final CNN training using MOGEO-selected hyperparameters.")
    parser.add_argument("--dataset-dir", default=config.DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hparams-file", default=None,
                         help=f"Path to a JSON file (default: results/{config.BEST_HPARAMS_FILENAME}) "
                              f"produced by run_mogeo.py. Falls back to BASELINE_HPARAMS if missing.")
    parser.add_argument("--epochs", type=int, default=config.FINAL_MAX_EPOCHS,
                         help=f"Hard maximum epochs (default/spec max: {config.FINAL_MAX_EPOCHS}).")
    parser.add_argument("--patience", type=int, default=config.FINAL_EARLY_STOP_PATIENCE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.epochs > config.FINAL_MAX_EPOCHS:
        print(f"NOTE: requested --epochs {args.epochs} exceeds the project spec cap of "
              f"{config.FINAL_MAX_EPOCHS}; clamping to {config.FINAL_MAX_EPOCHS}.")
        args.epochs = config.FINAL_MAX_EPOCHS

    set_seed(args.seed)
    splits_dir = os.path.join(args.output_dir, config.DEFAULT_SPLITS_DIR)
    checkpoint_dir = os.path.join(args.output_dir, config.DEFAULT_CHECKPOINT_DIR)
    results_dir = os.path.join(args.output_dir, config.DEFAULT_RESULTS_DIR)
    config.ensure_dirs(splits_dir, checkpoint_dir, results_dir)

    df = build_manifest(args.dataset_dir, splits_dir, seed=args.seed)
    print(leakage_report(df.to_dict("records")))

    device = get_device()
    print(f"Device: {device}")

    hparams_path = args.hparams_file or os.path.join(results_dir, config.BEST_HPARAMS_FILENAME)
    if os.path.exists(hparams_path):
        payload = load_json(hparams_path)
        hparams = config.HParams.from_dict(payload["hparams"])
        print(f"Loaded MOGEO-selected hyperparameters from {hparams_path}: {hparams.to_dict()}")
        source = "mogeo"
    else:
        hparams = config.BASELINE_HPARAMS
        print(f"No MOGEO hyperparameter file found at {hparams_path}; "
              f"falling back to BASELINE_HPARAMS: {hparams.to_dict()}")
        source = "baseline_fallback"

    best_ckpt = os.path.join(checkpoint_dir, config.FINAL_CHECKPOINT_NAME)  # best.pt
    last_ckpt = os.path.join(checkpoint_dir, config.FINAL_CHECKPOINT_RESUME_NAME)
    log_path = os.path.join(results_dir, config.EXPERIMENT_LOG_FILENAME)

    print(f"Final training: max_epochs={args.epochs} patience={args.patience} hparams_source={source}")

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

    plot_training_curves(result["history"], os.path.join(results_dir, "training_curves.png"))

    append_experiment_log(log_path, {
        "run_id": "final",
        "stage": "final",
        "epochs_trained": result["epochs_trained"],
        "hparams": hparams.to_dict(),
        "best_val_acc": round(result["best_val_acc"], 4),
        "best_val_macro_f1": round(result["best_val_macro_f1"], 4),
        "final_val_loss": round(result["history"]["val_loss"][-1], 4),
        "notes": f"hparams_source={source}, num_params={result['num_params']}, "
                 f"stopped_early={result['stopped_early']}, wall_time_s={dt:.1f}",
    })

    print(f"\nFinal training done in {dt:.1f}s | best_val_acc={result['best_val_acc']:.4f} "
          f"best_val_macro_f1={result['best_val_macro_f1']:.4f} (epoch {result['best_epoch']}/{result['epochs_trained']})")
    print(f"best.pt saved to: {best_ckpt}")
    print("Next step: run evaluate_final.py to evaluate ONCE on the untouched final_test split.")


if __name__ == "__main__":
    main()
