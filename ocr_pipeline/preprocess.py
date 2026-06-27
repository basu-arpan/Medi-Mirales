"""
OpenCV preprocessing pipeline for handwritten/scanned prescription images.

Pipeline stages, in order:
    1. Grayscale conversion
    2. Noise reduction        (fastNlMeansDenoising — preserves text edges
                                 better than a plain Gaussian blur on handwriting)
    3. Deskew                  (corrects camera-tilt / crooked scans via
                                 minAreaRect on the text mass)
    4. Adaptive binarization   (Otsu + adaptive threshold hybrid — handles
                                 uneven lighting across a photographed page
                                 better than a single global threshold)
    5. Border cleanup + light dilation to reconnect broken pen strokes

Each stage is saved to disk so the frontend can show the *actual* image at
each step — this is the trust-building "show your work" view for clinic staff
who are validating that the system isn't silently mangling a prescription.
"""

import os
import time

import cv2
import numpy as np


def _save_stage(img, output_dir, record_id, stage_name):
    fname = f"{record_id}_{stage_name}.png"
    path = os.path.join(output_dir, fname)
    cv2.imwrite(path, img)
    return f"/static/processed/{fname}"


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Estimate and correct page/text skew.

    Approach: threshold to isolate ink, find all foreground pixel
    coordinates, fit a minimum-area rotated rectangle around them, and use
    that rectangle's angle as the skew estimate. This is more robust than
    Hough-line skew detection on handwriting, since cursive strokes rarely
    produce long straight lines for Hough to lock onto.
    """
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        # Not enough ink detected to safely estimate an angle — skip deskew
        return gray

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns angles in [-90, 0); normalize to a small
    # rotation around 0 rather than accidentally flipping the page
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip correction for negligible skew to avoid introducing resample blur
    if abs(angle) < 0.3:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def _adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Hybrid binarization: adaptive Gaussian threshold handles uneven
    lighting/shadows across a photographed prescription pad, then a light
    Otsu pass on the result cleans residual speckle without losing thin
    pen strokes the way a single aggressive global threshold would.
    """
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=35,
        C=11,
    )
    return adaptive


def preprocess_image(input_path: str, output_dir: str, record_id: str) -> dict:
    """
    Run the full preprocessing pipeline on a raw uploaded image.

    Returns a dict with per-stage preview URLs (for the frontend pipeline
    visualization), per-stage timings in ms, and the final cleaned image
    path that gets handed to Tesseract.
    """
    timings_ms = {}
    stage_previews = {}

    t0 = time.time()
    original = cv2.imread(input_path)
    if original is None:
        raise ValueError("Could not read image — file may be corrupted or in an unsupported format.")

    # Downscale extremely large phone-camera photos for speed; upscale tiny
    # ones so Tesseract has enough pixel resolution per character stroke.
    h, w = original.shape[:2]
    target_width = 1600
    if w != target_width:
        scale = target_width / w
        original = cv2.resize(original, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    timings_ms["load_and_normalize"] = round((time.time() - t0) * 1000, 1)

    # --- Stage 1: Grayscale ---
    t0 = time.time()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    timings_ms["grayscale"] = round((time.time() - t0) * 1000, 1)
    stage_previews["1_grayscale"] = _save_stage(gray, output_dir, record_id, "1_grayscale")

    # --- Stage 2: Noise reduction ---
    t0 = time.time()
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    timings_ms["denoise"] = round((time.time() - t0) * 1000, 1)
    stage_previews["2_denoised"] = _save_stage(denoised, output_dir, record_id, "2_denoised")

    # --- Stage 3: Deskew ---
    t0 = time.time()
    deskewed = _deskew(denoised)
    timings_ms["deskew"] = round((time.time() - t0) * 1000, 1)
    stage_previews["3_deskewed"] = _save_stage(deskewed, output_dir, record_id, "3_deskewed")

    # --- Stage 4: Contrast normalization (CLAHE) ---
    # Evens out lighting gradients (shadow from a phone camera, faded ink)
    # before binarization, which meaningfully reduces false-black regions.
    t0 = time.time()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_normalized = clahe.apply(deskewed)
    timings_ms["contrast_normalize"] = round((time.time() - t0) * 1000, 1)
    stage_previews["4_contrast"] = _save_stage(contrast_normalized, output_dir, record_id, "4_contrast")

    # --- Stage 5: Adaptive binarization ---
    t0 = time.time()
    binarized = _adaptive_binarize(contrast_normalized)
    timings_ms["binarize"] = round((time.time() - t0) * 1000, 1)
    stage_previews["5_binarized"] = _save_stage(binarized, output_dir, record_id, "5_binarized")

    # --- Stage 6: Morphological cleanup ---
    # binarized is already black-text-on-white (THRESH_BINARY). Dilation
    # needs to grow the dark ink, which means operating on the inverted
    # image (where ink=255), then inverting back — operating directly on
    # `binarized` would instead grow the white background and erode strokes.
    t0 = time.time()
    kernel = np.ones((2, 2), np.uint8)
    inverted_for_morph = cv2.bitwise_not(binarized)  # ink=255, background=0
    dilated = cv2.dilate(inverted_for_morph, kernel, iterations=1)
    cleaned = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))
    final = cv2.bitwise_not(cleaned)  # back to black-text-on-white for Tesseract + human review
    timings_ms["morphological_cleanup"] = round((time.time() - t0) * 1000, 1)
    stage_previews["6_final"] = _save_stage(final, output_dir, record_id, "6_final")

    final_path = os.path.join(output_dir, f"{record_id}_6_final.png")

    return {
        "final_path": final_path,
        "final_preview": stage_previews["6_final"],
        "stage_previews": stage_previews,
        "timings_ms": timings_ms,
    }
