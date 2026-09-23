"""Shared utilities: seeding, checkpointing, logging, and leakage checks."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_md5(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Checkpointing (resumable across Colab disconnects)
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, state: Dict) -> None:
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX and Windows NTFS


def load_checkpoint(path: str, map_location=None) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopper:
    """Stops training when a monitored metric stops improving.

    Assumes higher is better (e.g. validation accuracy / macro-F1).
    """

    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True if `value` is the new best."""
        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

    def state_dict(self) -> Dict:
        return {
            "best": self.best,
            "counter": self.counter,
            "should_stop": self.should_stop,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.best = state["best"]
        self.counter = state["counter"]
        self.should_stop = state["should_stop"]
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)


# ---------------------------------------------------------------------------
# Experiment logging (append-only CSV, deliverable #13)
# ---------------------------------------------------------------------------
EXPERIMENT_LOG_FIELDS = [
    "timestamp",
    "run_id",
    "stage",
    "generation",
    "candidate_id",
    "epochs_trained",
    "hparams",
    "best_val_acc",
    "best_val_macro_f1",
    "final_val_loss",
    "notes",
]


def append_experiment_log(log_path: str, row: Dict) -> None:
    file_exists = os.path.exists(log_path)
    row = {**{k: "" for k in EXPERIMENT_LOG_FIELDS}, **row}
    row["timestamp"] = row["timestamp"] or time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Leakage checks (mandatory: final_test must never leak into dev)
# ---------------------------------------------------------------------------
def assert_no_overlap(*groups: Sequence[str], names: Optional[Sequence[str]] = None) -> None:
    """Raises AssertionError if any two groups of file paths intersect."""
    names = names or [f"group_{i}" for i in range(len(groups))]
    sets = [set(g) for g in groups]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                sample = list(overlap)[:5]
                raise AssertionError(
                    f"Data leakage detected between '{names[i]}' and '{names[j]}': "
                    f"{len(overlap)} overlapping files, e.g. {sample}"
                )


def leakage_report(manifest_rows: Iterable[Dict]) -> str:
    """Human-readable summary confirming dev/test disjointness, for logs."""
    by_split: Dict[str, List[str]] = {}
    for row in manifest_rows:
        by_split.setdefault(row["split"], []).append(row["filepath"])

    lines = ["Leakage check report", "=" * 40]
    splits = list(by_split.keys())
    for s in splits:
        lines.append(f"{s}: {len(by_split[s])} files")

    try:
        assert_no_overlap(*by_split.values(), names=splits)
        lines.append("RESULT: PASS -- no overlapping files between any splits.")
    except AssertionError as e:
        lines.append(f"RESULT: FAIL -- {e}")
    return "\n".join(lines)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
