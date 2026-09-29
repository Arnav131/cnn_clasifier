"""Central configuration: paths, seeds, and the MOGEO hyperparameter search space.

All paths default to locations relative to the repository root ("code/") so
the pipeline works unchanged locally and on Google Colab -- on Colab you
simply pass --dataset-dir / --output-dir pointing into your mounted Drive
folder (see instructions.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

# ---------------------------------------------------------------------------
# Default paths (all relative to the directory the scripts are invoked from,
# i.e. the "code" project root). Every script exposes these as CLI flags.
# ---------------------------------------------------------------------------
DEFAULT_DATASET_DIR = "dataset"
DEFAULT_PREPROCESSED_DIR = "preprocessed_dataset"
DEFAULT_FEATURES_FILENAME = "extracted_features.csv"
DEFAULT_OUTPUT_DIR = "."
DEFAULT_SPLITS_DIR = "splits"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_RESULTS_DIR = "results"

MANIFEST_FILENAME = "manifest.csv"
CLASSES_FILENAME = "classes.json"
DUPLICATE_REPORT_FILENAME = "duplicate_report.txt"

EXPERIMENT_LOG_FILENAME = "experiment_log.csv"
MOGEO_RESULTS_FILENAME = "mogeo_results.csv"
MOGEO_STATE_FILENAME = "mogeo_state.pkl"
BEST_HPARAMS_FILENAME = "best_hparams.json"

BASELINE_CHECKPOINT_NAME = "baseline_best.pt"
FINAL_CHECKPOINT_NAME = "best.pt"
FINAL_CHECKPOINT_RESUME_NAME = "final_last.pt"

# ---------------------------------------------------------------------------
# Global reproducibility / split configuration
# ---------------------------------------------------------------------------
SEED = 42

# Fraction of the FULL dataset held out as the untouched final test set.
FINAL_TEST_FRACTION = 0.15
# Fraction of the remaining DEV set (85%) held out as the internal
# validation set used for early stopping, baseline evaluation, and every
# MOGEO candidate's fitness evaluation. Never touches final_test.
DEV_VAL_FRACTION = 0.20

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# ---------------------------------------------------------------------------
# Fixed architecture / training constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 150
IMAGE_SIZE = 128  # matches the input resolution used by the model
NUM_WORKERS = 0 if os.name == "nt" else 2  # 0 on Windows prevents spawn multiprocessing friction

# Epoch guidance (per project spec -- do NOT blindly force max epochs)
MOGEO_CANDIDATE_EPOCHS = 10
FINAL_MAX_EPOCHS = 40
FINAL_EARLY_STOP_PATIENCE = 7
BASELINE_EPOCHS = 20
BASELINE_EARLY_STOP_PATIENCE = 5


@dataclass
class HParams:
    """A single point in the search space MOGEO explores.

    Decoded from a normalized [0, 1]^d vector -- see mogeo.py HyperParamSpace.
    Includes CNN architecture, training knobs, and feature selection flags.
    """

    num_blocks: int = 3
    base_channels: int = 32
    dropout: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    label_smoothing: float = 0.0
    use_features: bool = True
    feature_proj_dim: int = 32
    select_shape: bool = True
    select_size: bool = True
    select_colour: bool = True
    select_vein: bool = True
    select_texture: bool = True

    def to_dict(self) -> Dict:
        return {
            "num_blocks": self.num_blocks,
            "base_channels": self.base_channels,
            "dropout": round(float(self.dropout), 4),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "batch_size": self.batch_size,
            "label_smoothing": round(float(self.label_smoothing), 4),
            "use_features": self.use_features,
            "feature_proj_dim": self.feature_proj_dim,
            "select_shape": self.select_shape,
            "select_size": self.select_size,
            "select_colour": self.select_colour,
            "select_vein": self.select_vein,
            "select_texture": self.select_texture,
        }

    @staticmethod
    def from_dict(d: Dict) -> "HParams":
        return HParams(
            num_blocks=int(d["num_blocks"]),
            base_channels=int(d["base_channels"]),
            dropout=float(d["dropout"]),
            lr=float(d["lr"]),
            weight_decay=float(d["weight_decay"]),
            batch_size=int(d["batch_size"]),
            label_smoothing=float(d.get("label_smoothing", 0.0)),
            use_features=bool(d.get("use_features", True)),
            feature_proj_dim=int(d.get("feature_proj_dim", 32)),
            select_shape=bool(d.get("select_shape", True)),
            select_size=bool(d.get("select_size", True)),
            select_colour=bool(d.get("select_colour", True)),
            select_vein=bool(d.get("select_vein", True)),
            select_texture=bool(d.get("select_texture", True)),
        )


# Fixed baseline hyperparameters: standard CNN architecture with AdamW,
# and all 5 feature groups enabled.
BASELINE_HPARAMS = HParams(
    num_blocks=3,
    base_channels=32,
    dropout=0.5,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    label_smoothing=0.0,
    use_features=True,
    feature_proj_dim=32,
    select_shape=True,
    select_size=True,
    select_colour=True,
    select_vein=True,
    select_texture=True,
)


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)

