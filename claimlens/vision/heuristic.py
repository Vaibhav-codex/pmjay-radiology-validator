"""Classical-CV fallback backend (no model weights needed).

DEMO-GRADE: these are interpretable image-processing heuristics so the whole pipeline
runs anywhere. They are NOT clinically validated. For real use plug in
TorchXRayVision / MONAI / YOLO backends (see backends.py) and calibrate on labelled data.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import blob_log
from skimage.filters import apply_hysteresis_threshold, sato, threshold_otsu
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import convex_hull_image, skeletonize

from ..schemas import Finding
from .base import VisionBackend, VisionResult

SIZE = 768


def _resize(g):
    h, w = g.shape
    s = SIZE / max(h, w)
    return cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else g


def _bbox_norm(ys, xs, h, w, pad=0.01):
    return [round(max(0, xs.min() / w - pad), 4), round(max(0, ys.min() / h - pad), 4),
            round(min(1, (xs.max() + 1) / w + pad), 4), round(min(1, (ys.max() + 1) / h + pad), 4)]


def _remove_small(mask, min_px):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


# ------------------------------------------------------------------ chest
def _mirror(mask, cx):
    """Mirror a boolean mask about the vertical line x = cx."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    xs = np.arange(w)
    xm = np.round(2 * cx - xs).astype(int)
    ok = (xm >= 0) & (xm < w)
    out[:, xm[ok]] = mask[:, xs[ok]]
    return out


def _lungs(g):
    """Estimate the two lung fields. The healthy side is mirrored onto the other so that
    consolidations / effusions that were thresholded OUT of the dark mask are recovered."""
    h, w = g.shape
    sigma = max(h, w) / 150
    blur = cv2.GaussianBlur(g, (0, 0), sigma).astype(np.float32)
    roi = np.zeros((h, w), bool)
    y_lo, y_hi = int(.05 * h), int(.95 * h)
    roi[y_lo:y_hi, int(.08 * w):int(.92 * w)] = True
    thr = threshold_otsu(blur[roi])
    m = ndi.binary_closing(ndi.binary_opening((blur < thr) & roi, iterations=2), iterations=3)
    lab, n = ndi.label(m)
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 0.02 * h * w:
            continue
        if (ys.max() - ys.min()) > 0.85 * (y_hi - y_lo):      # background strips, not lungs
            continue
        comps.append((len(ys), xs.mean() / w, i, xs.mean()))
    left = [c for c in comps if c[1] < 0.5]
    right = [c for c in comps if c[1] >= 0.5]
    if not left or not right:
        return None
    dark_l = np.isin(lab, [c[2] for c in left])
    dark_r = np.isin(lab, [c[2] for c in right])
    xl = np.average([c[3] for c in left], weights=[c[0] for c in left])
    xr = np.average([c[3] for c in right], weights=[c[0] for c in right])
    cx = (xl + xr) / 2
    if not (0.25 * w < xr - xl < 0.75 * w):                     # lungs must be a plausible distance apart
        return None
    shrink = max(2, int(2.5 * sigma))
    fields = {"L": ndi.binary_erosion(convex_hull_image(dark_l | _mirror(dark_r, cx)), iterations=shrink),
              "R": ndi.binary_erosion(convex_hull_image(dark_r | _mirror(dark_l, cx)), iterations=shrink)}
    dark = dark_l | dark_r
    ref = float(np.median(blur[dark]))
    sig = float(1.4826 * np.median(np.abs(blur[dark] - ref)))
    return blur, {"L": (dark_l, fields["L"]), "R": (dark_r, fields["R"])}, ref, sig


def _chest(g):
    h, w = g.shape
    res = _lungs(g)
    if res is None:
        return [], None
    blur, lungs, ref, sig = res
    op_thr = ref + max(4 * sig, 18)
    findings = []
    for side, (dark, hull) in lungs.items():
        lat = "right" if side == "L" else "left"          # patient's right lung is on the viewer's LEFT
        opaque = hull & (blur > op_thr)
        hull_area = max(hull.sum(), 1)
        # ---- effusion: homogeneous opacity rising from the costophrenic angle
        row_hull = hull.sum(axis=1)
        row_op = opaque.sum(axis=1)
        rows = np.nonzero(row_hull > 0)[0]
        y_top, y_bot = rows.min(), rows.max()
        level = 0
        for y in range(y_bot, y_top, -1):
            if row_hull[y] > 0 and row_op[y] / row_hull[y] > 0.55:
                level += 1
            elif row_hull[y] > 0:
                break
        level_frac = level / max(1, (y_bot - y_top))
        if level_frac >= 0.08:
            band = np.zeros_like(opaque)
            band[y_bot - level:y_bot + 1] = True
            ys, xs = np.nonzero(opaque & band)
            if len(ys):
                sev = "mild" if level_frac < 0.15 else "moderate" if level_frac < 0.33 else "severe"
                conf = float(min(0.92, 0.55 + level_frac * 1.6))
                findings.append(Finding("effusion", round(conf, 2), _bbox_norm(ys, xs, h, w), lat, sev,
                                        f"Basal homogeneous opacity occupying ~{level_frac:.0%} of {lat} hemithorax height",
                                        dict(fill_fraction=round(float(level_frac), 3))))
            opaque = opaque & ~band
        # ---- focal mass vs diffuse infiltrate
        opaque = ndi.binary_opening(opaque, iterations=2)
        frac = float(opaque.sum() / hull_area)
        lab_o = sk_label(opaque)
        props = [p for p in regionprops(lab_o) if p.area >= 0.004 * hull_area]
        if props:
            big = max(props, key=lambda p: p.area)
            f_big = big.area / hull_area
            if len(props) <= 2 and f_big < 0.12 and big.solidity > 0.85 and big.eccentricity < 0.85:
                y0, x0, y1, x1 = big.bbox
                diam = max(y1 - y0, x1 - x0) / w
                sev = "mild" if f_big < 0.02 else "moderate" if f_big < 0.06 else "severe"
                findings.append(Finding("mass", round(float(min(0.88, 0.5 + big.solidity * 0.3 + f_big * 2)), 2),
                                        [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)], lat, sev,
                                        f"Well-defined focal opacity (~{diam:.0%} of image width)",
                                        dict(area_fraction=round(float(f_big), 3), solidity=round(float(big.solidity), 2))))
            elif frac >= 0.12:
                ys, xs = np.nonzero(opaque)
                sev = "mild" if frac < 0.22 else "moderate" if frac < 0.38 else "severe"
                findings.append(Finding("infiltrate", round(float(min(0.9, 0.5 + (frac - 0.12) * 2.0)), 2),
                                        _bbox_norm(ys, xs, h, w), lat, sev,
                                        f"Diffuse increased opacity involving ~{frac:.0%} of {lat} lung field",
                                        dict(involvement=round(frac, 3))))
    return findings, "chest"


# ------------------------------------------------------------------ bone
def _bone(g):
    h, w = g.shape
    blur = cv2.GaussianBlur(g, (0, 0), 2)
    mask = blur > threshold_otsu(blur)
    mask = _remove_small(mask, int(0.01 * h * w))
    if mask.sum() < 0.02 * h * w:
        return [], None
    region = ndi.binary_closing(mask, iterations=max(4, int(0.015 * max(h, w))))
    inner = ndi.binary_erosion(region, iterations=2)
    ridge = sato(g.astype(np.float32) / 255.0, sigmas=(1.5, 2.5), black_ridges=True)
    vals = ridge[inner]
    thr = max(0.09, 0.5 * float(vals.max()))
    cand = ndi.binary_opening((ridge > thr) & inner, structure=np.ones((2, 2)))
    lab = sk_label(cand)
    ys_b, xs_b = np.nonzero(region)
    bone_span = max(ys_b.max() - ys_b.min(), xs_b.max() - xs_b.min())
    segs = []
    for p in regionprops(lab):
        if p.axis_major_length >= 0.04 * max(h, w) and p.axis_major_length / max(p.axis_minor_length, 1) >= 2.5:
            strength = float(ridge[lab == p.label].mean())
            segs.append((p, strength))
    if not segs:
        return [], "bone"
    findings = []
    n = len(segs)
    # hysteresis: follow weak continuations of the strong lines to measure total fracture-line length
    hyst = apply_hysteresis_threshold(np.where(inner, ridge, 0.0), 0.05, thr)
    hyst = ndi.binary_opening(hyst, structure=np.ones((2, 2)))
    skel_len = float(skeletonize(hyst).sum())
    bone_width = region.sum() / max(bone_span, 1)
    ratio = skel_len / max(bone_width, 1)          # total fracture-line length in units of bone width
    sev = "severe" if ratio >= 1.35 else "moderate" if ratio >= 1.1 else "mild"   # tunable
    if hyst.sum() > 0:
        ys, xs = np.nonzero(hyst)                 # every connected fracture-line pixel
    else:
        ys = np.concatenate([np.nonzero(lab == p.label)[0] for p, _ in segs])
        xs = np.concatenate([np.nonzero(lab == p.label)[1] for p, _ in segs])
    strength = float(np.mean([s for _, s in segs]))
    conf = float(min(0.9, 0.5 + strength * 3.0 + 0.05 * n))
    desc = ("Multiple intersecting lucent lines / fragments - pattern suggests comminution" if sev == "severe"
            else "Lucent line crossing the cortex - consistent with fracture line")
    findings.append(Finding("fracture", round(conf, 2), _bbox_norm(ys, xs, h, w, 0.03), "unknown", sev, desc,
                            dict(fracture_segments=n, line_length_ratio=round(ratio, 2), ridge_strength=round(strength, 3))))
    return findings, "bone"


# ------------------------------------------------------------------ generic (US / CT / MRI)
def _generic(g, modality):
    h, w = g.shape
    blur = cv2.GaussianBlur(g, (0, 0), 3).astype(np.float32)
    findings = []
    fg = blur > 12
    filled = ndi.binary_fill_holes(ndi.binary_closing(fg, iterations=3))
    if modality == "ultrasound":
        holes = filled & ~fg
        holes = ndi.binary_opening(holes, iterations=3)
        for p in regionprops(sk_label(holes)):
            frac = p.area / max(filled.sum(), 1)
            if 0.01 <= frac <= 0.45:
                m = sk_label(holes) == p.label
                inside_std = float(g[m].std())
                if inside_std < 14:                       # anechoic / homogeneous
                    y0, x0, y1, x1 = p.bbox
                    sev = "mild" if frac < 0.04 else "moderate" if frac < 0.15 else "severe"
                    findings.append(Finding("fluid_collection", round(float(min(0.88, 0.55 + frac * 2)), 2),
                                            [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
                                            "unknown", sev, f"Anechoic collection (~{frac:.0%} of scan area)",
                                            dict(area_fraction=round(float(frac), 3))))
    f = blur / 255.0
    blobs = blob_log(f, min_sigma=w * 0.012, max_sigma=w * 0.06, num_sigma=6, threshold=0.04)
    cands = []
    for y, x, s in blobs:
        r = s * 1.414
        yy, xx = np.ogrid[:h, :w]
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        inner = d2 <= r ** 2
        ring = (d2 > (1.4 * r) ** 2) & (d2 <= (1.9 * r) ** 2)
        if ring.sum() < 20 or not filled[int(y), int(x)] or filled[ring].mean() < 0.9:
            continue
        contrast = float(f[inner].mean() - f[ring].mean())
        if contrast > 0.12:
            cands.append((contrast, y, x, r))
    cands.sort(reverse=True)
    for contrast, y, x, r in cands[:2]:
        frac = (np.pi * r * r) / (h * w)
        findings.append(Finding("mass", round(float(min(0.85, 0.45 + contrast * 1.5)), 2),
                                [round(max(0, x - r) / w, 4), round(max(0, y - r) / h, 4),
                                 round(min(w, x + r) / w, 4), round(min(h, y + r) / h, 4)],
                                "unknown", "mild" if frac < 0.01 else "moderate" if frac < 0.04 else "severe",
                                f"Focal hyper-attenuating/echogenic lesion, contrast +{contrast:.2f}",
                                dict(contrast=round(contrast, 3))))
    return findings, "other"


def guess_body_part(g) -> str:
    return "chest" if _lungs(_resize(g)) is not None else "long_bone"


class HeuristicBackend(VisionBackend):
    name = "heuristic-cv"
    detectable = {"fracture", "infiltrate", "effusion", "mass", "fluid_collection"}

    def analyze(self, gray, modality, body_part):
        g = _resize(gray)
        if body_part == "auto":
            body_part = guess_body_part(gray)
        notes = []
        if modality in ("xray", "auto") and body_part == "chest":
            findings, _ = _chest(g)
        elif modality in ("xray", "auto") and body_part in ("long_bone", "hand", "knee"):
            findings, _ = _bone(g)
        else:
            findings, _ = _generic(g, modality)
            notes.append("Generic focal-lesion heuristics used for this modality/body part.")
        return VisionResult(findings, None, self.detectable, notes)
