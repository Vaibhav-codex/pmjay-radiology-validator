"""Explainability: heatmaps + bounding boxes + labels for the human adjudicator."""
import cv2
import numpy as np

COLORS = {"fracture": (0, 165, 255), "infiltrate": (255, 200, 0), "effusion": (255, 120, 60),
          "mass": (60, 60, 255), "fluid_collection": (200, 100, 255)}


def heat_from_findings(shape, findings):
    h, w = shape
    heat = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[:h, :w]
    for f in findings:
        if not f.bbox:
            continue
        x0, y0, x1, y1 = f.bbox[0] * w, f.bbox[1] * h, f.bbox[2] * w, f.bbox[3] * h
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        sx, sy = max((x1 - x0) / 2.5, 6), max((y1 - y0) / 2.5, 6)
        heat = np.maximum(heat, f.confidence * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2))
    return heat


def render(gray, findings, heat=None, max_side=1400):
    h, w = gray.shape
    s = min(1.0, max_side / max(h, w))
    g = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else gray
    h, w = g.shape
    base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)
    if heat is None:
        heat = heat_from_findings((h, w), findings)
    elif heat.shape != (h, w):
        heat = cv2.resize(heat, (w, h))
    if heat.max() > 0.05:
        hm = cv2.applyColorMap((np.clip(heat / max(heat.max(), 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        alpha = np.clip(heat, 0, 1)[..., None] * 0.65
        base = base * (1 - alpha) + hm.astype(np.float32) * alpha
    out = base.astype(np.uint8)
    for f in findings:
        if not f.bbox:
            continue
        x0, y0, x1, y1 = int(f.bbox[0] * w), int(f.bbox[1] * h), int(f.bbox[2] * w), int(f.bbox[3] * h)
        col = COLORS.get(f.label, (0, 255, 0))
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        txt = f"{f.label} {f.confidence:.2f}" + (f" ({f.laterality})" if f.laterality != "unknown" else "")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 4, y0)
        cv2.rectangle(out, (x0, ty - th - 4), (x0 + tw + 6, ty + 2), col, -1)
        cv2.putText(out, txt, (x0 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def save_png(path, bgr_or_gray):
    cv2.imwrite(str(path), bgr_or_gray)
