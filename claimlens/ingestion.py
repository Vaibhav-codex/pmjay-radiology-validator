"""Ingest claim packages: DICOM / JPEG / PNG images and TXT / PDF / scanned-image reports."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from .nlp import ocr

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path) -> np.ndarray:
    """Return an 8-bit grayscale array."""
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        try:
            import pydicom
        except ImportError as e:
            raise RuntimeError("DICOM support needs `pip install pydicom`") from e
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3:                       # multi-frame / volume -> middle slice
            arr = arr[arr.shape[0] // 2]
        slope, inter = float(getattr(ds, "RescaleSlope", 1)), float(getattr(ds, "RescaleIntercept", 0))
        arr = arr * slope + inter
        lo, hi = np.percentile(arr, [0.5, 99.5])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            arr = 1 - arr
        return (arr * 255).astype(np.uint8)
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def load_report(path=None, text: str | None = None):
    """Return (text, ocr_confidence|None, source)."""
    if text and text.strip():
        return text.strip(), None, "pasted"
    if not path:
        return "", None, "none"
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(errors="ignore"), None, "text-file"
    if ext == ".pdf":
        return ocr.read_pdf(path)
    if ext in IMAGE_EXT:
        t, conf = ocr.ocr_image(load_image(path))
        return t, conf, "ocr-image"
    raise ValueError(f"Unsupported report format: {ext}")
