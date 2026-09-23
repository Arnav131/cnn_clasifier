"""Dataset manifest construction, leakage-safe splitting, and DataLoaders.

The manifest is built ONCE and persisted to disk (splits/manifest.csv). Every
downstream script (baseline, MOGEO, final training, final evaluation) reads
the SAME manifest, so the untouched final_test split is guaranteed to be
identical and never re-randomized between runs -- this is what the legacy
notebook got wrong (it called train_test_split with different parameters in
almost every cell).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config
from .utils import file_md5

MANIFEST_COLUMNS = ["filepath", "label", "class_idx", "split", "file_hash"]


def _list_images(dataset_dir: str) -> List[Tuple[str, str]]:
    """Returns list of (relative_filepath, class_label) for every image."""
    items = []
    for class_name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if os.path.splitext(fname)[1].lower() in config.IMAGE_EXTENSIONS:
                rel_path = os.path.join(class_name, fname)
                items.append((rel_path, class_name))
    return items


def _fix_duplicate_leakage(rows: List[Dict], log_lines: List[str]) -> None:
    """Ensures byte-identical duplicate images never span dev/test splits.

    Any duplicate group that spans final_test (or dev_val) and any other
    split is collapsed entirely into dev_train, since a duplicate held out
    for evaluation whose twin was used in training would silently leak
    information into that evaluation. This mutates `rows` in place.
    """
    by_hash: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_hash[row["file_hash"]].append(row)

    n_groups_fixed = 0
    n_files_moved = 0
    for h, group in by_hash.items():
        if len(group) < 2:
            continue
        splits_present = {r["split"] for r in group}
        if len(splits_present) > 1:
            n_groups_fixed += 1
            for r in group:
                if r["split"] != "dev_train":
                    n_files_moved += 1
                r["split"] = "dev_train"
            log_lines.append(
                f"  Duplicate group (hash={h[:10]}...): {len(group)} files spanned "
                f"{sorted(splits_present)} -> all reassigned to dev_train. "
                f"Files: {[r['filepath'] for r in group]}"
            )

    log_lines.append(
        f"\nSummary: {n_groups_fixed} duplicate groups touched eval splits; "
        f"{n_files_moved} file(s) reassigned to dev_train to prevent leakage."
    )


def build_manifest(
    dataset_dir: str,
    splits_dir: str,
    seed: int = config.SEED,
    final_test_fraction: float = config.FINAL_TEST_FRACTION,
    dev_val_fraction: float = config.DEV_VAL_FRACTION,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Builds (or loads a cached) manifest with dev_train/dev_val/final_test splits.

    This function is idempotent: once splits/manifest.csv exists it is loaded
    as-is (unless force_rebuild=True), guaranteeing every script in the
    pipeline operates on an identical, fixed partition of the data.
    """
    os.makedirs(splits_dir, exist_ok=True)
    manifest_path = os.path.join(splits_dir, config.MANIFEST_FILENAME)
    classes_path = os.path.join(splits_dir, config.CLASSES_FILENAME)
    dup_report_path = os.path.join(splits_dir, config.DUPLICATE_REPORT_FILENAME)

    if os.path.exists(manifest_path) and not force_rebuild:
        df = pd.read_csv(manifest_path)
        return df

    items = _list_images(dataset_dir)
    if not items:
        raise RuntimeError(f"No images found under '{dataset_dir}'.")

    classes = sorted({label for _, label in items})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    labels = [label for _, label in items]

    # Stage 1: carve out the untouched final_test split (stratified).
    train_val_idx, test_idx = train_test_split(
        list(range(len(items))),
        test_size=final_test_fraction,
        stratify=labels,
        random_state=seed,
    )

    # Stage 2: split the remaining dev set into dev_train / dev_val.
    dev_labels = [labels[i] for i in train_val_idx]
    dev_train_idx, dev_val_idx = train_test_split(
        train_val_idx,
        test_size=dev_val_fraction,
        stratify=dev_labels,
        random_state=seed,
    )

    split_of_index: Dict[int, str] = {}
    for i in dev_train_idx:
        split_of_index[i] = "dev_train"
    for i in dev_val_idx:
        split_of_index[i] = "dev_val"
    for i in test_idx:
        split_of_index[i] = "final_test"

    rows: List[Dict] = []
    for i, (rel_path, label) in enumerate(items):
        abs_path = os.path.join(dataset_dir, rel_path)
        rows.append(
            {
                "filepath": rel_path,
                "label": label,
                "class_idx": class_to_idx[label],
                "split": split_of_index[i],
                "file_hash": file_md5(abs_path),
            }
        )

    log_lines = ["Duplicate / leakage-fix report", "=" * 40]
    _fix_duplicate_leakage(rows, log_lines)

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df.to_csv(manifest_path, index=False)

    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    with open(dup_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    return df


def load_classes(splits_dir: str) -> List[str]:
    classes_path = os.path.join(splits_dir, config.CLASSES_FILENAME)
    with open(classes_path, encoding="utf-8") as f:
        return json.load(f)


def get_transforms(split: str, image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    if split == "dev_train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class PlantDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_dir: str, split: str, image_size: int = config.IMAGE_SIZE):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = get_transforms(split, image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        abs_path = os.path.join(self.dataset_dir, row["filepath"])
        image = Image.open(abs_path).convert("RGB")
        image = self.transform(image)
        return image, int(row["class_idx"]), row["filepath"]


def get_dataloader(
    df: pd.DataFrame,
    dataset_dir: str,
    split: str,
    batch_size: int,
    image_size: int = config.IMAGE_SIZE,
    num_workers: int = 2,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    dataset = PlantDataset(df, dataset_dir, split, image_size)
    if shuffle is None:
        shuffle = split == "dev_train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
