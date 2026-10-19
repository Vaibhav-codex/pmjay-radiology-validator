"""End-to-end orchestration: ingest -> preprocess -> dual-stream extraction -> correlation -> STG -> report."""
import json
import re
import time
from pathlib import Path

import cv2

from . import config, ingestion, preprocessing
from .dedupe import Registry
from .fusion import discrepancy, scoring
from .nlp.entities import RuleBasedExtractor, highlight_html
from .ontology import CANONICAL, display
from .schemas import ClaimMeta, Flag, ImageAnalysis, to_dict
from .stg import validator
from .vision import explain
from .vision.backends import get_backend
from .vision.heuristic import guess_body_part

_extractor = None


def _safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


def analyze_claim(meta: ClaimMeta, image_paths, report_path=None, report_text=None,
                  out_root=None, registry=None, extractor=None, commit=True):
    global _extractor
    extractor = extractor or _extractor or RuleBasedExtractor()
    _extractor = extractor
    rules = validator.load_rules()
    out_dir = Path(out_root or config.OUTPUT_DIR) / _safe(meta.claim_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = registry or Registry()
    t0 = time.time()
    flags, analyses, hashes, detectable = [], [], [], set()

    # ---------------- vision stream ----------------
    for i, p in enumerate(image_paths):
        p = Path(p)
        image_id = f"img{i + 1}"
        raw = ingestion.load_image(p)
        gray = preprocessing.crop_borders(raw)
        quality = preprocessing.assess_quality(gray)
        enhanced = preprocessing.enhance(gray, quality)

        modality = meta.modality
        body = meta.body_part if meta.body_part != "auto" else (guess_body_part(enhanced) if meta.modality in ("auto", "xray") else "other")
        if modality == "auto":
            modality = "xray"
        backend = get_backend(modality, body)
        res = backend.analyze(enhanced, modality, body)
        detectable |= res.detectable
        for f in res.findings:
            f.image_id = image_id
            # low-quality images reduce confidence (never raise it)
            f.confidence = round(f.confidence * (0.6 + 0.4 * quality["score"]), 2)
        res.findings = [f for f in res.findings if f.confidence >= 0.2]

        h, w = enhanced.shape
        explain.save_png(out_dir / f"{image_id}_original.png", gray)
        explain.save_png(out_dir / f"{image_id}_enhanced.png", enhanced)
        explain.save_png(out_dir / f"{image_id}_explain.png", explain.render(enhanced, res.findings, res.heatmap))
        analyses.append(ImageAnalysis(
            image_id=image_id, filename=p.name, modality=modality, body_part=body, backend=backend.name,
            quality=quality, findings=res.findings,
            artifacts={k: f"{_safe(meta.claim_id)}/{image_id}_{k}.png" for k in ("original", "enhanced", "explain")}))

        # repeated / recycled image check
        hsh, hits = registry.check_image(meta.claim_id, p.name, enhanced)
        hashes.append((p.name, hsh))
        other = [h_ for h_ in hits if h_["claim_id"] != meta.claim_id]
        same = [h_ for h_ in hits if h_["claim_id"] == meta.claim_id]
        if other:
            flags.append(Flag("RECYCLED_IMAGE", "high", f"Image {p.name} already used in claim {other[0]['claim_id']}",
                              "Near-identical radiograph (perceptual hash match) was submitted under a different claim ID - possible image reuse.",
                              dict(matches=other)))
        elif same:
            flags.append(Flag("DUPLICATE_SUBMISSION", "info", f"Image {p.name} was submitted before for this claim",
                              "Repeated submission of the same image under the same claim.", dict(matches=same)))
    if len(analyses) > 1:   # duplicates inside this submission
        for a in range(len(hashes)):
            for b in range(a + 1, len(hashes)):
                from .dedupe import hamming
                if hamming(hashes[a][1], hashes[b][1]) <= config.T["dup_hamming"]:
                    flags.append(Flag("DUPLICATE_SUBMISSION", "info", "Duplicate images inside this submission",
                                      f"{hashes[a][0]} and {hashes[b][0]} are near-identical."))

    # ---------------- text stream ----------------
    text, ocr_conf, source = ingestion.load_report(report_path, report_text)
    entities = extractor.extract(text, ocr_conf)
    rhash, rhits = registry.check_report(meta.claim_id, text)
    if [r for r in rhits if r["claim_id"] != meta.claim_id]:
        flags.append(Flag("DUPLICATE_SUBMISSION", "medium", "Identical report text used in another claim",
                          "Report text matches a report already submitted under a different claim ID.",
                          dict(matches=[r["claim_id"] for r in rhits])))
    if not text.strip():
        flags.append(Flag("NOT_VERIFIABLE", "medium", "No report text supplied", "Cannot cross-verify without a hospital report."))
    if ocr_conf is not None and ocr_conf < 0.6:
        flags.append(Flag("POOR_IMAGE_QUALITY", "low", "Low OCR confidence on report scan",
                          f"Mean OCR confidence {ocr_conf:.0%}; extracted entities may be incomplete."))

    # ---------------- fusion ----------------
    img_best = discrepancy.aggregate_image_findings(analyses)
    d_flags, alignment = discrepancy.detect(img_best, entities, meta, analyses, detectable)
    flags += d_flags

    # ---------------- STG ----------------
    rule_results, pkg = validator.validate(meta, img_best, entities, analyses, detectable, rules)

    score = scoring.risk_score(flags, rule_results)
    rec = scoring.recommendation(score, flags, rule_results)
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    flags.sort(key=lambda f: order[f.severity])

    result = dict(
        claim=to_dict(meta),
        package=dict(code=meta.package_code, **({k: pkg[k] for k in ("name", "specialty", "notes") if k in pkg} if pkg else {})),
        images=[to_dict(a) for a in analyses],
        report=dict(source=source, ocr_confidence=ocr_conf, extractor=entities.extractor, sites=entities.sites,
                    conservative_days=entities.conservative_days,
                    findings=[to_dict(f) for f in entities.findings if True],
                    procedures=[to_dict(p) for p in entities.procedures],
                    highlighted_html=highlight_html(entities)),
        image_findings_aggregated={k: dict(to_dict(v), display=display(v.label), snomed=CANONICAL.get(k, {}).get("snomed"),
                                           icd10=CANONICAL.get(k, {}).get("icd10")) for k, v in img_best.items()},
        alignment_score=alignment,
        flags=[to_dict(f) for f in flags],
        stg_checks=[to_dict(r) for r in rule_results],
        risk_score=score,
        recommendation=rec,
        meta=dict(runtime_sec=round(time.time() - t0, 2), rules_version=rules.get("version"), created=time.strftime("%Y-%m-%d %H:%M:%S")),
    )
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    if commit:
        registry.commit(meta.claim_id, hashes, rhash)
    return result
