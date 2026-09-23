#!/usr/bin/env python3
"""MOGEO hyperparameter/architecture search (outer optimizer).

Each candidate is trained with AdamW for a small, fixed epoch budget
(~10 epochs, guidance value -- see config.MOGEO_CANDIDATE_EPOCHS) on
dev_train, evaluated on dev_val. final_test is NEVER touched here.

Resumable: MOGEO state (population, archive, RNG) is checkpointed after
every single candidate evaluation, so a Colab disconnect loses at most one
in-progress candidate. Re-running this script with --resume continues from
where it left off.

Usage:
    python run_mogeo.py --dataset-dir dataset --output-dir . --resume
"""
import argparse
import os
import time

import pandas as pd

from pipeline import config
from pipeline.data import build_manifest
from pipeline.evaluate import plot_mogeo_convergence
from pipeline.mogeo import MOGEO
from pipeline.train import train_one_run
from pipeline.utils import (
    append_experiment_log,
    get_device,
    leakage_report,
    save_json,
    set_seed,
)

MOGEO_RESULTS_FIELDS = [
    "generation", "eagle_idx", "candidate_id", "val_acc", "val_macro_f1",
    "epochs_trained", "num_params", "num_blocks", "base_channels", "dropout",
    "lr", "weight_decay", "batch_size", "label_smoothing",
]


def append_mogeo_result(path: str, row: dict) -> None:
    import csv
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MOGEO_RESULTS_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="MOGEO search over CNN hyperparameters.")
    parser.add_argument("--dataset-dir", default=config.DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--candidate-epochs", type=int, default=config.MOGEO_CANDIDATE_EPOCHS)
    parser.add_argument("--candidate-patience", type=int, default=4)
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

    device = get_device()
    print(f"Device: {device}")

    log_path = os.path.join(results_dir, config.EXPERIMENT_LOG_FILENAME)
    mogeo_results_path = os.path.join(results_dir, config.MOGEO_RESULTS_FILENAME)
    mogeo_state_path = os.path.join(checkpoint_dir, config.MOGEO_STATE_FILENAME)

    def objective_fn(hparams, candidate_id):
        t0 = time.time()
        result = train_one_run(
            hparams=hparams,
            df=df,
            dataset_dir=args.dataset_dir,
            device=device,
            max_epochs=args.candidate_epochs,
            patience=args.candidate_patience,
            best_checkpoint_path=None,
            last_checkpoint_path=None,
            resume=False,
            num_workers=args.num_workers,
            seed=args.seed,
            verbose=False,
            save_weights=False,
        )
        dt = time.time() - t0
        extra = {"epochs_trained": result["epochs_trained"], "num_params": result["num_params"], "wall_time_s": dt}
        print(f"  [{candidate_id}] {hparams.to_dict()} -> "
              f"val_acc={result['best_val_acc']:.4f} val_f1={result['best_val_macro_f1']:.4f} "
              f"({result['epochs_trained']} epochs, {dt:.1f}s)")
        return result["best_val_acc"], result["best_val_macro_f1"], extra

    def on_candidate_evaluated(generation, eagle_idx, ind):
        append_mogeo_result(mogeo_results_path, {
            "generation": generation,
            "eagle_idx": eagle_idx,
            "candidate_id": ind.candidate_id,
            "val_acc": round(ind.objectives[0], 4),
            "val_macro_f1": round(ind.objectives[1], 4),
            "epochs_trained": ind.extra.get("epochs_trained"),
            "num_params": ind.extra.get("num_params"),
            **{k: ind.hparams[k] for k in ["num_blocks", "base_channels", "dropout", "lr",
                                            "weight_decay", "batch_size", "label_smoothing"]},
        })
        append_experiment_log(log_path, {
            "run_id": ind.candidate_id,
            "stage": "mogeo",
            "generation": generation,
            "candidate_id": ind.candidate_id,
            "epochs_trained": ind.extra.get("epochs_trained"),
            "hparams": ind.hparams,
            "best_val_acc": round(ind.objectives[0], 4),
            "best_val_macro_f1": round(ind.objectives[1], 4),
            "notes": f"wall_time_s={ind.extra.get('wall_time_s', 0):.1f}",
        })

    if not args.resume and os.path.exists(mogeo_state_path):
        print(f"WARNING: existing MOGEO state found at {mogeo_state_path} but --resume was not "
              f"passed. Delete this file (or pass --resume) before starting a new search.")
        return

    optimizer = MOGEO(
        objective_fn=objective_fn,
        pop_size=args.pop_size,
        generations=args.generations,
        seed=args.seed,
        state_path=mogeo_state_path,
        on_candidate_evaluated=on_candidate_evaluated,
    )

    print(f"Running MOGEO: pop_size={args.pop_size} generations={args.generations} "
          f"candidate_epochs={args.candidate_epochs} "
          f"(total candidate trainings <= {args.pop_size * args.generations})")

    archive = optimizer.run()

    best = optimizer.best_compromise()
    print(f"\nBest compromise solution: {best.hparams}")
    print(f"  val_acc={best.objectives[0]:.4f} val_macro_f1={best.objectives[1]:.4f}")

    best_hparams_path = os.path.join(results_dir, config.BEST_HPARAMS_FILENAME)
    save_json(best_hparams_path, {
        "hparams": best.hparams,
        "val_acc": best.objectives[0],
        "val_macro_f1": best.objectives[1],
        "candidate_id": best.candidate_id,
        "archive_size": len(archive),
        "pop_size": args.pop_size,
        "generations": args.generations,
        "candidate_epochs": args.candidate_epochs,
    })
    print(f"Best hyperparameters saved to: {best_hparams_path}")

    mogeo_df = pd.read_csv(mogeo_results_path)
    plot_mogeo_convergence(mogeo_df, os.path.join(results_dir, "mogeo_convergence.png"))
    print(f"MOGEO convergence plot saved to: {os.path.join(results_dir, 'mogeo_convergence.png')}")


if __name__ == "__main__":
    main()
