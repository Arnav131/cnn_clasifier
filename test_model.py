#!/usr/bin/env python3
"""Standalone script to load best.pt and run predictions.

This is the script referenced in instructions.md for testing best.pt after
downloading it from Google Drive/Colab -- e.g. to feed it back into your
local Antigravity/Zed session for further inspection.

Usage:
    # Single image
    python test_model.py --checkpoint checkpoints/best.pt --image path/to/leaf.jpg

    # Folder of images
    python test_model.py --checkpoint checkpoints/best.pt --dir path/to/folder

    # Full evaluation against the untouched final_test split (prints accuracy only;
    # use evaluate_final.py for the full report/confusion matrix/error analysis)
    python test_model.py --checkpoint checkpoints/best.pt --eval-test --dataset-dir dataset
"""
import argparse
import os

import torch
from PIL import Image

from pipeline import config
from pipeline.data import build_manifest, get_dataloader, get_transforms, load_classes
from pipeline.evaluate import load_model_from_checkpoint, run_inference, compute_metrics
from pipeline.utils import get_device


def load_properties(properties_csv: str):
    if not os.path.exists(properties_csv):
        return {}
    import pandas as pd

    df = pd.read_csv(properties_csv, encoding="latin1")
    return {row["common_name"]: row.to_dict() for _, row in df.iterrows()}


def predict_image(model, image_path: str, classes, device, top_k: int = 5):
    transform = get_transforms("final_test", image_size=config.IMAGE_SIZE)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_idx = torch.topk(probs, k=min(top_k, len(classes)))
    return [(classes[i], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]


def print_prediction(image_path: str, predictions, properties: dict):
    print(f"\n{image_path}")
    for rank, (label, prob) in enumerate(predictions, start=1):
        print(f"  #{rank}: {label:30s} {prob*100:5.1f}%")
    top_label = predictions[0][0]
    if top_label in properties:
        info = properties[top_label]
        print(f"  -> Botanical name: {info.get('botanical_name', 'Unknown')}")
        print(f"  -> Family: {info.get('family', 'Unknown')}")
        print(f"  -> Medicinal property: {info.get('medicinal_property', 'Unknown')}")
        print(f"  -> Side effects: {info.get('side_effects', 'Unknown')}")


def main():
    parser = argparse.ArgumentParser(description="Test best.pt on images or the final_test split.")
    parser.add_argument("--checkpoint", default=os.path.join(config.DEFAULT_CHECKPOINT_DIR, config.FINAL_CHECKPOINT_NAME))
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-dir", default=config.DEFAULT_DATASET_DIR)
    parser.add_argument("--properties-csv", default="properties.csv")
    parser.add_argument("--image", default=None, help="Path to a single image to classify.")
    parser.add_argument("--dir", default=None, help="Path to a folder of images to classify.")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate on the untouched final_test split.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    splits_dir = os.path.join(args.output_dir, config.DEFAULT_SPLITS_DIR)
    classes = load_classes(splits_dir)
    device = get_device()

    model, ckpt = load_model_from_checkpoint(args.checkpoint, num_classes=len(classes), device=device)
    print(f"Loaded {args.checkpoint} | trained to epoch {ckpt.get('epoch')} | "
          f"dev_val acc at save time = {ckpt.get('val_acc', float('nan')):.4f}")

    if args.image is None and args.dir is None and not args.eval_test:
        print("Nothing to do -- pass --image, --dir, or --eval-test. See --help.")
        return

    if args.image or args.dir:
        properties = load_properties(args.properties_csv)

    if args.image:
        preds = predict_image(model, args.image, classes, device, args.top_k)
        print_prediction(args.image, preds, properties)

    if args.dir:
        for fname in sorted(os.listdir(args.dir)):
            if os.path.splitext(fname)[1].lower() not in config.IMAGE_EXTENSIONS:
                continue
            path = os.path.join(args.dir, fname)
            preds = predict_image(model, path, classes, device, args.top_k)
            print_prediction(path, preds, properties)

    if args.eval_test:
        df = build_manifest(args.dataset_dir, splits_dir)
        loader = get_dataloader(df, args.dataset_dir, "final_test", batch_size=32, shuffle=False)
        y_true, y_pred, _, _ = run_inference(model, loader, device)
        metrics = compute_metrics(y_true, y_pred, classes)
        print(f"\nfinal_test accuracy: {metrics['accuracy']*100:.2f}% "
              f"({metrics['num_correct']}/{metrics['num_samples']})")
        print(f"final_test macro_f1: {metrics['macro_f1']*100:.2f}%")


if __name__ == "__main__":
    main()
