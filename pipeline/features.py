"""Feature extraction module implementing Shape, Size, Colour, Vein, and Texture.

Directly implements the 'Feature Extraction' stage of the plant classification flowchart:
  - Shape: Area, Perimeter, Circularity, Aspect Ratio, Solidity, Extent, Eccentricity
  - Size: Bounding Box Dimensions, Leaf Area Ratio, Equivalent Diameter
  - Colour: RGB and HSV channel means and standard deviations within the leaf mask
  - Vein: Interior Canny edge density, Gabor filter responses (4 angles), Top-Hat vein ridges
  - Texture: GLCM features (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM)
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


FEATURE_GROUPS = ["shape", "size", "colour", "vein", "texture"]

FEATURE_NAMES = [
    # Shape (7)
    "shape_contour_area",
    "shape_perimeter",
    "shape_circularity",
    "shape_aspect_ratio",
    "shape_solidity",
    "shape_extent",
    "shape_eccentricity",
    # Size (4)
    "size_bbox_width",
    "size_bbox_height",
    "size_leaf_area_ratio",
    "size_equiv_diameter",
    # Colour (12)
    "colour_r_mean",
    "colour_r_std",
    "colour_g_mean",
    "colour_g_std",
    "colour_b_mean",
    "colour_b_std",
    "colour_h_mean",
    "colour_h_std",
    "colour_s_mean",
    "colour_s_std",
    "colour_v_mean",
    "colour_v_std",
    # Vein (7)
    "vein_edge_density",
    "vein_gabor_0",
    "vein_gabor_45",
    "vein_gabor_90",
    "vein_gabor_135",
    "vein_tophat_mean",
    "vein_blackhat_mean",
    # Texture (6)
    "texture_contrast",
    "texture_dissimilarity",
    "texture_homogeneity",
    "texture_energy",
    "texture_correlation",
    "texture_asm",
]


def extract_shape_features(mask: np.ndarray) -> Dict[str, float]:
    """Extracts geometric shape features from the leaf binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {k: 0.0 for k in FEATURE_NAMES if k.startswith("shape_")}

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    circularity = (4.0 * math.pi * area) / (perimeter * perimeter + 1e-6)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / float(h + 1e-6)

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / (hull_area + 1e-6)

    bbox_area = float(w * h)
    extent = area / (bbox_area + 1e-6)

    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (cx, cy), (ma, ma_major), angle = cv2.fitEllipse(cnt)
            a = max(ma, ma_major) / 2.0
            b = min(ma, ma_major) / 2.0
            if a > 0:
                eccentricity = math.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))
        except Exception:
            eccentricity = 0.0

    return {
        "shape_contour_area": area,
        "shape_perimeter": perimeter,
        "shape_circularity": circularity,
        "shape_aspect_ratio": aspect_ratio,
        "shape_solidity": solidity,
        "shape_extent": extent,
        "shape_eccentricity": eccentricity,
    }


def extract_size_features(mask: np.ndarray) -> Dict[str, float]:
    """Extracts size and dimension metrics from the leaf binary mask."""
    total_pixels = float(mask.shape[0] * mask.shape[1])
    leaf_pixels = float(np.count_nonzero(mask > 0))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {k: 0.0 for k in FEATURE_NAMES if k.startswith("size_")}

    cnt = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(cnt)
    area = float(cv2.contourArea(cnt))
    equiv_diameter = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0

    return {
        "size_bbox_width": float(w),
        "size_bbox_height": float(h),
        "size_leaf_area_ratio": leaf_pixels / max(total_pixels, 1.0),
        "size_equiv_diameter": equiv_diameter,
    }


def extract_colour_features(image_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extracts mean and standard deviation for RGB and HSV within the leaf mask."""
    leaf_pixels = mask > 0
    if not np.any(leaf_pixels):
        return {k: 0.0 for k in FEATURE_NAMES if k.startswith("colour_")}

    r = image_rgb[:, :, 0][leaf_pixels]
    g = image_rgb[:, :, 1][leaf_pixels]
    b = image_rgb[:, :, 2][leaf_pixels]

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0][leaf_pixels]
    s = hsv[:, :, 1][leaf_pixels]
    v = hsv[:, :, 2][leaf_pixels]

    return {
        "colour_r_mean": float(np.mean(r)),
        "colour_r_std": float(np.std(r)),
        "colour_g_mean": float(np.mean(g)),
        "colour_g_std": float(np.std(g)),
        "colour_b_mean": float(np.mean(b)),
        "colour_b_std": float(np.std(b)),
        "colour_h_mean": float(np.mean(h)),
        "colour_h_std": float(np.std(h)),
        "colour_s_mean": float(np.mean(s)),
        "colour_s_std": float(np.std(s)),
        "colour_v_mean": float(np.mean(v)),
        "colour_v_std": float(np.std(v)),
    }


def extract_vein_features(image_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extracts leaf vein characteristics: internal edge density, Gabor filter responses, Top-Hat."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    leaf_pixels = mask > 0
    leaf_count = float(np.count_nonzero(leaf_pixels))

    if leaf_count < 10:
        return {k: 0.0 for k in FEATURE_NAMES if k.startswith("vein_")}

    # Canny edge detector for internal venation
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edges_in_leaf = edges[leaf_pixels]
    edge_density = float(np.count_nonzero(edges_in_leaf > 0)) / leaf_count

    # Gabor filters at 4 orientations (0, 45, 90, 135 deg)
    thetas = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]
    gabor_keys = ["vein_gabor_0", "vein_gabor_45", "vein_gabor_90", "vein_gabor_135"]
    gabor_results = {}
    for theta, key in zip(thetas, gabor_keys):
        kernel = cv2.getGaborKernel((9, 9), sigma=2.0, theta=theta, lambd=6.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        fimg = cv2.filter2D(gray, cv2.CV_32F, kernel)
        resp = np.abs(fimg)[leaf_pixels]
        gabor_results[key] = float(np.mean(resp)) if len(resp) > 0 else 0.0

    # Morphological Top-Hat / Black-Hat (highlights bright and dark vein ridges)
    kernel_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_structure)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_structure)
    tophat_mean = float(np.mean(tophat[leaf_pixels])) if leaf_count > 0 else 0.0
    blackhat_mean = float(np.mean(blackhat[leaf_pixels])) if leaf_count > 0 else 0.0

    return {
        "vein_edge_density": edge_density,
        **gabor_results,
        "vein_tophat_mean": tophat_mean,
        "vein_blackhat_mean": blackhat_mean,
    }


def extract_texture_features(image_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extracts Gray-Level Co-occurrence Matrix (GLCM) texture descriptors."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    masked_gray = np.where(mask > 0, gray, 0).astype(np.uint8)

    # GLCM with distances=1 and angles=[0, pi/4, pi/2, 3pi/4]
    try:
        glcm = graycomatrix(
            masked_gray,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )
        contrast = float(np.mean(graycoprops(glcm, "contrast")))
        dissimilarity = float(np.mean(graycoprops(glcm, "dissimilarity")))
        homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
        energy = float(np.mean(graycoprops(glcm, "energy")))
        correlation = float(np.mean(graycoprops(glcm, "correlation")))
        asm = float(np.mean(graycoprops(glcm, "ASM")))
    except Exception:
        contrast = dissimilarity = homogeneity = energy = correlation = asm = 0.0

    return {
        "texture_contrast": contrast,
        "texture_dissimilarity": dissimilarity,
        "texture_homogeneity": homogeneity,
        "texture_energy": energy,
        "texture_correlation": correlation,
        "texture_asm": asm,
    }


def extract_all_features(image_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extracts all 36 quantitative features across Shape, Size, Colour, Vein, and Texture."""
    features: Dict[str, float] = {}
    features.update(extract_shape_features(mask))
    features.update(extract_size_features(mask))
    features.update(extract_colour_features(image_rgb, mask))
    features.update(extract_vein_features(image_rgb, mask))
    features.update(extract_texture_features(image_rgb, mask))
    return features


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Converts a features dict to an ordered float32 numpy array matching FEATURE_NAMES."""
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
