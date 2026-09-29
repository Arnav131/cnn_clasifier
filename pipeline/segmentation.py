"""Image segmentation module for leaf foreground isolation.

Implements the 'Image Segmentation' stage of the plant classification flowchart.
Segments the leaf from background using color masking, Otsu thresholding,
morphological cleanup, and contour-guided GrabCut refinement.
"""
from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def segment_leaf(image_rgb: np.ndarray, use_grabcut: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Segments the leaf from background.

    Args:
        image_rgb: HxWx3 uint8 RGB image.
        use_grabcut: Whether to refine the mask with GrabCut (defaults to True).

    Returns:
        segmented_rgb: HxWx3 uint8 image with background masked out to black (0).
        mask: HxW uint8 binary mask where 255 is leaf foreground, 0 is background.
    """
    h, w = image_rgb.shape[:2]
    if h == 0 or w == 0:
        return image_rgb, np.ones((h, w), dtype=np.uint8) * 255

    # 1. Color space conversion
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # 2. Compute Excess Green Index (ExG = 2*G - R - B)
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu thresholding on ExG
    _, mask_otsu = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # HSV green / plant range mask
    lower_green = np.array([20, 25, 25])
    upper_green = np.array([105, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_green, upper_green)

    # Combined initial mask
    mask_combined = cv2.bitwise_or(mask_otsu, mask_hsv)

    # 3. Morphological operations (remove noise, fill interior holes)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_small, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # 4. Find largest connected component (the leaf)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback to entire image if no contours found
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return image_rgb, mask

    largest_cnt = max(contours, key=cv2.contourArea)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(leaf_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)

    # Fill internal holes of the largest contour
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # 5. Optional GrabCut refinement
    if use_grabcut and cv2.contourArea(largest_cnt) > 200:
        try:
            x, y, bw, bh = cv2.boundingRect(largest_cnt)
            margin = 5
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(w, x + bw + margin)
            y1 = min(h, y + bh + margin)
            rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

            grab_mask = np.zeros((h, w), dtype=np.uint8)
            grab_mask[leaf_mask == 255] = cv2.GC_PR_FGD
            grab_mask[leaf_mask == 0] = cv2.GC_BGD

            if rect[2] > 10 and rect[3] > 10:
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                cv2.grabCut(image_rgb, grab_mask, rect, bgd_model, fgd_model, iterCount=3, mode=cv2.GC_INIT_WITH_MASK)
                final_mask = np.where((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                if np.sum(final_mask) > 100:
                    leaf_mask = final_mask
        except Exception:
            pass

    segmented_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=leaf_mask)
    return segmented_rgb, leaf_mask


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (128, 128)) -> Tuple[np.ndarray, np.ndarray]:
    """Loads image, applies bilateral smoothing, segments the leaf, and resizes.

    Returns:
        segmented_rgb: (target_size[0], target_size[1], 3) uint8 RGB array
        mask: (target_size[0], target_size[1]) uint8 binary mask array
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    # Mild bilateral filtering to preserve edges while smoothing noise
    filtered = cv2.bilateralFilter(arr, d=5, sigmaColor=50, sigmaSpace=50)

    segmented, mask = segment_leaf(filtered, use_grabcut=True)

    # Resize to target size
    segmented_resized = cv2.resize(segmented, target_size, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    return segmented_resized, mask_resized
