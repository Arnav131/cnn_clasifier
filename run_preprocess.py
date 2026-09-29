#!/usr/bin/env python3
"""Image Preprocessing, Segmentation, and Quantitative Feature Extraction.

Directly implements flowchart steps 3, 4, and 5:
  - Step 3: Dataset Preparation (ensures manifest is ready)
  - Step 4: Image Preprocessing (resizing, color normalization, noise reduction)
  - Step 5: Image Segmentation (leaf foreground isolation via HSV masking + GrabCut)
  - Step 6: Feature Extraction (Shape, Size, Colour, Vein, Texture)

Outputs:
  - preprocessed_dataset/<class>/<image>: segmented leaf RGB images
  - extracted_features.csv: quantitative tabular features for each image

Usage:
  python run_preprocess.py --dataset-dir dataset --output-dir .
"""
import argparse
import concurrent.futures
import os
import time
from typing import Dict, List, Optional

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

from pipeline import config
from pipeline.data import build_manifest
from pipeline.features import FEATURE_NAMES, extract_all_features
from pipeline.segmentation import segment_leaf


def process_single_image(
    rel_path: str,
    dataset_dir: str,
    preprocessed_dir: str,
    target_size: int = config.IMAGE_SIZE,
    save_segmented: bool = True,
) -> Optional[Dict]:
    src_path = os.path.join(dataset_dir, rel_path)
    if not os.path.exists(src_path):
        return None

    try:
        # 1. Load image
        img = Image.open(src_path).convert("RGB")
        arr = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB) if hasattr(cv2, "imread") else None
        if arr is None:
            import numpy as np
            arr = np.array(img)

        # 2. Preprocess & Segment leaf foreground
        filtered = cv2.bilateralFilter(arr, d=5, sigmaColor=50, sigmaSpace=50)
        segmented_rgb, mask = segment_leaf(filtered, use_grabcut=True)

        # 3. Resize to target size
        segmented_resized = cv2.resize(segmented_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        # 4. Save segmented image if requested
        if save_segmented:
            dst_path = os.path.join(preprocessed_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            # Save using PIL
            out_img = Image.fromarray(segmented_resized)
            # If original had webp or other extension, save accordingly
            out_img.save(dst_path, quality=95)

        # 5. Extract quantitative features: Shape, Size, Colour, Vein, Texture
        feats = extract_all_features(segmented_resized, mask_resized)
        norm_path = rel_path.replace("\\", "/")
        return {"filepath": norm_path, **feats}
    except Exception as e:
        print(f"Warning: failed to process {rel_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Preprocessing, Segmentation, and Feature Extraction.")
    parser.add_argument("--dataset-dir", default=config.DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preprocessed-dir", default=config.DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--features-csv", default=config.DEFAULT_FEATURES_FILENAME)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--skip-existing-csv", action="store_true",
                        help="Skip extraction if features CSV already exists.")
    args = parser.parse_args()

    splits_dir = os.path.join(args.output_dir, config.DEFAULT_SPLITS_DIR)
    preprocessed_full_dir = os.path.join(args.output_dir, args.preprocessed_dir)
    features_full_path = os.path.join(args.output_dir, args.features_csv)
    config.ensure_dirs(splits_dir, preprocessed_full_dir)

    print("=== Step 1 & 2: Dataset Preparation & Leakage Check ===")
    df = build_manifest(args.dataset_dir, splits_dir)
    print(f"Total manifest entries: {len(df)}")

    if args.skip_existing_csv and os.path.exists(features_full_path):
        existing_df = pd.read_csv(features_full_path)
        print(f"Features CSV already exists at {features_full_path} with {len(existing_df)} rows.")
        return

    print("\n=== Step 3, 4 & 5: Preprocessing, Segmentation & Feature Extraction ===")
    print(f"Input: {args.dataset_dir}")
    print(f"Segmented images output: {preprocessed_full_dir}")
    print(f"Features output: {features_full_path}")
    print(f"Extracting 36 features: Shape (7), Size (4), Colour (12), Vein (7), Texture (6)...")

    file_list = df["filepath"].tolist()
    records: List[Dict] = []

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {
            executor.submit(
                process_single_image,
                rel_path,
                args.dataset_dir,
                preprocessed_full_dir,
                args.image_size,
                True,
            ): rel_path
            for rel_path in file_list
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(file_list), desc="Processing"):
            res = future.result()
            if res is not None:
                records.append(res)

    dt = time.time() - t0
    out_df = pd.DataFrame(records)
    # Order columns nicely: filepath, then FEATURE_NAMES
    cols = ["filepath"] + [c for c in FEATURE_NAMES if c in out_df.columns]
    out_df = out_df[cols]
    out_df.to_csv(features_full_path, index=False)

    print(f"\nCompleted in {dt:.1f}s ({dt/max(len(records), 1)*1000:.1f} ms/image)")
    print(f"Extracted {len(out_df)} records with {len(cols)-1} features to: {features_full_path}")
    print(f"Segmented leaf images saved to: {preprocessed_full_dir}")


if __name__ == "__main__":
    main()
