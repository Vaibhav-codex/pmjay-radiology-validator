"""Quality control + enhancement for real-world radiology images
(low contrast, noise, photographed films with borders, odd resolutions)."""
import cv2
import numpy as np
from . import config


def crop_borders(gray: np.ndarray) -> np.ndarray:
    """Trim near-uniform frames (e.g. a film photographed on a light box / phone screenshot)."""
    h, w = gray.shape
    rows = gray.std(axis=1) > 2.5
    cols = gray.std(axis=0) > 2.5
    if rows.sum() < 0.3 * h or cols.sum() < 0.3 * w:
        return gray
    y0, y1 = np.argmax(rows), h - np.argmax(rows[::-1])
    x0, x1 = np.argmax(cols), w - np.argmax(cols[::-1])
    if (y1 - y0) < 0.6 * h or (x1 - x0) < 0.6 * w:
        return gray
    return gray[y0:y1, x0:x1]


def assess_quality(gray: np.ndarray) -> dict:
    h, w = gray.shape
    s = 512 / min(h, w)
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    p1, p99 = np.percentile(gray, [1, 99])
    dyn = float(p99 - p1)
    mean = float(gray.mean())
    noise = float(np.median(np.abs(small[1:, :].astype(int) - small[:-1, :].astype(int))))
    flags = []
    if min(h, w) < config.T["min_side_px"]:
        flags.append("LOW_RESOLUTION")
    if dyn < config.T["min_dyn_range"]:
        flags.append("LOW_CONTRAST")
    if sharp < config.T["min_sharpness"]:
        flags.append("BLURRY")
    if mean < 30:
        flags.append("UNDEREXPOSED")
    elif mean > 215:
        flags.append("OVEREXPOSED")
    if noise > 14:
        flags.append("NOISY")
    score = 1.0 - 0.18 * len(flags)
    return dict(width=w, height=h, sharpness=round(sharp, 1), dynamic_range=round(dyn, 1),
                mean_intensity=round(mean, 1), noise=round(noise, 1), flags=flags,
                score=round(max(0.2, score), 2), acceptable=len(flags) <= 1)


def enhance(gray: np.ndarray, quality: dict | None = None) -> np.ndarray:
    """Denoise (only if noisy) + CLAHE contrast enhancement."""
    out = gray
    if quality and "NOISY" in quality["flags"]:
        out = cv2.fastNlMeansDenoising(out, None, h=9, templateWindowSize=7, searchWindowSize=21)
    else:
        out = cv2.GaussianBlur(out, (0, 0), 0.8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    out = clahe.apply(out)
    if quality and "LOW_RESOLUTION" in quality["flags"]:
        s = config.T["min_side_px"] / min(out.shape)
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return out
