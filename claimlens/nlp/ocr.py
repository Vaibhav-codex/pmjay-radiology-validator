"""OCR for scanned / photographed hospital reports (Tesseract; swap in PaddleOCR / LayoutLM if available)."""
import cv2
import numpy as np


def _deskew(bw: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(bw < 128))
    if len(coords) < 200:
        return bw
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 15:
        return bw
    h, w = bw.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def preprocess_for_ocr(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    if max(h, w) < 1800:
        s = 1800 / max(h, w)
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
    return _deskew(bw)


def ocr_image(gray: np.ndarray):
    """Returns (text, mean_word_confidence 0-1)."""
    try:
        import pytesseract
        from pytesseract import Output
        img = preprocess_for_ocr(gray)
        data = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 6")
        words, confs = [], []
        for t, c in zip(data["text"], data["conf"]):
            if t.strip() and float(c) >= 0:
                words.append(t)
                confs.append(float(c))
        text = pytesseract.image_to_string(img, config="--psm 6")
        return text, (float(np.mean(confs)) / 100 if confs else 0.0)
    except Exception as e:  # tesseract missing etc.
        return f"[OCR unavailable: {e}]", 0.0


def read_pdf(path):
    """Text PDF -> direct extraction; scanned PDF -> rasterise + OCR."""
    import pdfplumber
    text_parts, ocr_confs, used_ocr = [], [], False
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = (page.extract_text() or "").strip()
            if len(t) < 30:
                try:
                    img = np.array(page.to_image(resolution=200).original.convert("L"))
                    t, conf = ocr_image(img)
                    ocr_confs.append(conf)
                    used_ocr = True
                except Exception as e:
                    t = f"[could not rasterise scanned page: {e}]"
            text_parts.append(t)
    return "\n".join(text_parts), (float(np.mean(ocr_confs)) if ocr_confs else None), ("ocr-pdf" if used_ocr else "text-pdf")
