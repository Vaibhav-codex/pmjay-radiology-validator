"""Cross-modal correlation: compare image-derived findings with report-derived findings."""
from .. import config
from ..ontology import display
from ..schemas import Flag

R = config.SEVERITY_RANK
T = config.T


def aggregate_image_findings(analyses):
    """label -> best (highest-confidence) Finding across all submitted images."""
    best = {}
    for a in analyses:
        for f in a.findings:
            if f.label not in best or f.confidence > best[f.label].confidence:
                best[f.label] = f
    return best


def _lat_conflict(a, b):
    return a not in ("unknown", "bilateral") and b not in ("unknown", "bilateral") and a != b


def detect(img_best, report, meta, image_analyses, detectable):
    flags = []
    pos = [f for f in report.findings if not f.negated]
    neg = [f for f in report.findings if f.negated]
    pos_labels = {f.label for f in pos}
    neg_labels = {f.label for f in neg} - pos_labels
    poor_quality = any(not a.quality["acceptable"] for a in image_analyses)

    # ---- 1. omission / contradiction: image shows it, report silent or denies
    for label, f in img_best.items():
        if f.confidence < T["detect_conf"]:
            continue
        if label in neg_labels:
            flags.append(Flag("CONTRADICTION", "high", f"Report denies {display(label)} but image shows it",
                              f"The imaging model detected {display(label)} (confidence {f.confidence:.2f}) "
                              f"but the report explicitly negates it.", dict(label=label, confidence=f.confidence, bbox=f.bbox)))
        elif label not in pos_labels:
            flags.append(Flag("OMISSION", "high" if f.confidence >= 0.7 else "medium",
                              f"{display(label)} on image not mentioned in report",
                              f"{f.description or 'Anomaly detected'} (confidence {f.confidence:.2f}) with no corresponding "
                              f"statement in the hospital report.", dict(label=label, confidence=f.confidence, bbox=f.bbox)))

    # ---- 2. unsupported claim / exaggeration: report asserts, image doesn't back it
    seen = set()
    for rf in pos:
        if rf.label in seen:
            continue
        seen.add(rf.label)
        f = img_best.get(rf.label)
        if rf.label not in detectable:
            flags.append(Flag("NOT_VERIFIABLE", "info", f"{display(rf.label)} cannot be verified automatically",
                              "The active vision backend has no detector for this finding; adjudicator should check the image manually.",
                              dict(label=rf.label)))
            continue
        if f is None or f.confidence < T["weak_conf"]:
            sev = "high" if rf.severity in ("moderate", "severe") else "medium"
            why = " Image quality is poor, so absence is less certain." if poor_quality else ""
            flags.append(Flag("UNSUPPORTED_FINDING", sev, f"Report states {display(rf.label)} - not seen on image",
                              f'Report: "{rf.sentence[:140]}". No supporting image evidence was found.{why}',
                              dict(label=rf.label, report_severity=rf.severity)))
            continue
        gap = R[rf.severity] - R[f.severity] if f.severity != "unknown" else 0
        if gap >= 2:
            flags.append(Flag("EXAGGERATION", "high", f"{display(rf.label)} severity exaggerated",
                              f'Report describes it as "{rf.severity}" but image analysis indicates "{f.severity}". '
                              f'{f.description}', dict(label=rf.label, report=rf.severity, image=f.severity, bbox=f.bbox)))
        elif gap == 1:
            flags.append(Flag("POSSIBLE_OVERSTATEMENT", "low", f"{display(rf.label)} severity may be overstated",
                              f'Report "{rf.severity}" vs image "{f.severity}" - one grade apart, within measurement uncertainty.',
                              dict(label=rf.label, report=rf.severity, image=f.severity)))

    # ---- 3. laterality (report vs image vs billing vs film marker)
    for rf in pos:
        f = img_best.get(rf.label)
        if f and _lat_conflict(rf.laterality, f.laterality):
            flags.append(Flag("LATERALITY_MISMATCH", "high", f"Wrong side: report {rf.laterality}, image {f.laterality}",
                              f"{display(rf.label)} is reported on the {rf.laterality} but the image evidence is on the {f.laterality}.",
                              dict(label=rf.label, report=rf.laterality, image=f.laterality)))
    rep_lats = {f.laterality for f in pos if f.laterality not in ("unknown", "bilateral")}
    rep_lats |= {p.laterality for p in report.procedures if p.laterality not in ("unknown", "bilateral")}
    for src_name, src in (("billed procedure", meta.billed_laterality), ("film marker", meta.image_marker_laterality)):
        for rl in rep_lats:
            if _lat_conflict(src, rl):
                flags.append(Flag("LATERALITY_MISMATCH", "high", f"Wrong side: {src_name} {src}, report {rl}",
                                  f"The {src_name} says {src} but the report documents the {rl}.", dict(source=src_name, value=src, report=rl)))
    if _lat_conflict(meta.billed_laterality, meta.image_marker_laterality):
        flags.append(Flag("LATERALITY_MISMATCH", "high", "Wrong side: billed vs film marker",
                          f"Billed {meta.billed_laterality} but the film marker reads {meta.image_marker_laterality}."))
    for label, f in img_best.items():
        if f.confidence >= T["detect_conf"] and _lat_conflict(meta.billed_laterality, f.laterality):
            flags.append(Flag("LATERALITY_MISMATCH", "high", f"Wrong side: billed {meta.billed_laterality}, image {f.laterality}",
                              f"{display(label)} localised to the {f.laterality} side on the image.", dict(label=label)))

    # ---- 4. anatomical site consistency
    parts = {a.body_part for a in image_analyses}
    if meta.body_part != "auto" and report.sites:
        region_ok = meta.body_part in report.sites or (meta.body_part in ("long_bone", "hand", "knee") and set(report.sites) & {"long_bone", "hand", "knee"})
        if not region_ok:
            flags.append(Flag("BODY_PART_MISMATCH", "medium", "Report body region differs from submitted imaging",
                              f"Images are tagged '{meta.body_part}', report discusses {', '.join(report.sites)}.",
                              dict(image=meta.body_part, report=report.sites)))
    if "chest" in parts and report.sites and "chest" not in report.sites and any(f.label in ("effusion", "infiltrate") for f in pos):
        flags.append(Flag("BODY_PART_MISMATCH", "medium", "Chest image but report discusses another region",
                          f"Report sites: {', '.join(report.sites)}."))

    # ---- 5. image quality
    for a in image_analyses:
        if not a.quality["acceptable"]:
            flags.append(Flag("POOR_IMAGE_QUALITY", "low", f"Poor image quality: {a.filename}",
                              f"Quality issues: {', '.join(a.quality['flags'])}. Automated confidence is reduced - consider requesting a better scan.",
                              dict(image=a.filename, flags=a.quality["flags"])))

    # ---- alignment score (agreement over the union of findings)
    img_set = {l for l, f in img_best.items() if f.confidence >= T["detect_conf"]}
    union = img_set | pos_labels
    union = {l for l in union if l in detectable or l in img_set}
    alignment = 1.0 if not union else round(len(img_set & pos_labels & union) / len(union), 2)
    return flags, alignment
