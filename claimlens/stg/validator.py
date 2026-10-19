"""Deterministic STG / package rule engine (rules live in config/stg_rules.yaml)."""
from datetime import date
import yaml
from .. import config
from ..schemas import RuleResult

R = config.SEVERITY_RANK
T = config.T


def load_rules(path=None):
    return yaml.safe_load(open(path or config.STG_RULES_PATH))


def _d(s):
    try:
        return date.fromisoformat(s) if s else None
    except ValueError:
        return None


def validate(meta, img_best, report, image_analyses, detectable, rules):
    pkg = rules["packages"].get(meta.package_code)
    if not pkg:
        return [RuleResult("R00", "Package recognised", "fail", f"Unknown package code '{meta.package_code}'.")], None
    out = []
    req = pkg["required_findings"]
    pos = [f for f in report.findings if not f.negated and f.label in req]
    poor_q = any(not a.quality["acceptable"] for a in image_analyses)

    # R01 diagnosis documented
    out.append(RuleResult("R01", "Report documents the indicating diagnosis",
                          "pass" if pos else "fail",
                          f"Found: {', '.join(sorted({f.label for f in pos}))}." if pos else
                          f"Report does not affirm any of {req} required for '{pkg['name']}'."))

    # R02 imaging support
    imgs = [img_best[l] for l in req if l in img_best and img_best[l].confidence >= T["detect_conf"]]
    verifiable = any(l in detectable for l in req)
    need = pkg["image_support"]
    if imgs:
        out.append(RuleResult("R02", "Imaging supports the diagnosis", "pass",
                              "; ".join(f"{f.label} ({f.confidence:.2f}): {f.description}" for f in imgs)))
    elif not verifiable:
        out.append(RuleResult("R02", "Imaging supports the diagnosis", "warn",
                              "No automatic detector for this finding - manual image review needed."))
    else:
        st = "fail" if need == "required" and not poor_q else "warn"
        out.append(RuleResult("R02", "Imaging supports the diagnosis", st,
                              "No supporting finding on submitted images" + (" (image quality is poor)." if poor_q else ".")))

    # R03 severity justifies package tier
    ms = pkg.get("min_severity", "unknown")
    if ms != "unknown":
        isev = max([R[f.severity] for f in imgs], default=0)
        rsev = max([R[f.severity] for f in pos], default=0)
        if isev >= R[ms]:
            out.append(RuleResult("R03", f"Severity justifies tier (>= {ms})", "pass", "Image severity meets the minimum."))
        elif isev == 0 and not imgs:
            out.append(RuleResult("R03", f"Severity justifies tier (>= {ms})", "warn" if not verifiable else "fail",
                                  "No image-derived severity available."))
        elif rsev >= R[ms]:
            out.append(RuleResult("R03", f"Severity justifies tier (>= {ms})", "fail",
                                  "Report severity meets the threshold but image severity is lower - tier relies on report wording only."))
        else:
            out.append(RuleResult("R03", f"Severity justifies tier (>= {ms})", "fail", "Neither image nor report reaches the minimum severity."))
    else:
        out.append(RuleResult("R03", "Severity justifies tier", "na", "No minimum severity for this package."))

    # R04 modality
    mods = {a.modality for a in image_analyses}
    allowed = set(pkg["allowed_modalities"])
    out.append(RuleResult("R04", "Imaging modality appropriate",
                          "pass" if mods & allowed else "fail",
                          f"Submitted: {', '.join(sorted(mods))}; accepted: {', '.join(sorted(allowed))}."))

    # R05 body region
    parts = {a.body_part for a in image_analyses}
    ok = bool(parts & set(pkg["body_regions"])) or ("long_bone" in pkg["body_regions"] and parts & {"long_bone", "hand", "knee"}) \
        or (parts == {"other"})
    out.append(RuleResult("R05", "Imaged body region matches package", "pass" if ok else "fail",
                          f"Imaged: {', '.join(sorted(parts))}; package expects: {', '.join(pkg['body_regions'])}."))

    # R06 laterality
    if pkg.get("laterality_must_match"):
        srcs = {"billed": meta.billed_laterality, "film marker": meta.image_marker_laterality}
        srcs.update({f"report ({f.label})": f.laterality for f in pos})
        srcs.update({f"image ({f.label})": f.laterality for f in imgs})
        known = {k: v for k, v in srcs.items() if v not in ("unknown",)}
        vals = {v for v in known.values() if v != "bilateral"}
        if len(known) < 2:
            out.append(RuleResult("R06", "Laterality consistent", "na", "Not enough laterality information."))
        elif len(vals) <= 1:
            out.append(RuleResult("R06", "Laterality consistent", "pass", f"All sources agree ({', '.join(vals) or 'bilateral'})."))
        else:
            out.append(RuleResult("R06", "Laterality consistent", "fail", "Conflict: " + "; ".join(f"{k}={v}" for k, v in known.items())))
    else:
        out.append(RuleResult("R06", "Laterality consistent", "na", "Not required for this package."))

    # R07 conservative-management timeline
    need_days = pkg.get("conservative_days_min", 0)
    days = meta.conservative_days if meta.conservative_days is not None else report.conservative_days
    if need_days:
        if days is None:
            out.append(RuleResult("R07", f"Conservative management >= {need_days} days documented", "fail",
                                  "No conservative-management duration found in claim form or report."))
        elif days < need_days:
            out.append(RuleResult("R07", f"Conservative management >= {need_days} days documented", "fail",
                                  f"Only {days} days documented."))
        else:
            out.append(RuleResult("R07", f"Conservative management >= {need_days} days documented", "pass", f"{days} days documented."))
    else:
        out.append(RuleResult("R07", "Conservative management timeline", "na", "No pre-requisite trial for this package."))

    # R08 length of stay
    adm, dis = _d(meta.admission_date), _d(meta.discharge_date)
    los = pkg.get("los_days")
    if adm and dis and los:
        n = (dis - adm).days
        if n < 0:
            out.append(RuleResult("R08", "Length of stay within STG", "fail", "Discharge precedes admission."))
        elif los["min"] <= n <= los["max"]:
            out.append(RuleResult("R08", "Length of stay within STG", "pass", f"{n} days (allowed {los['min']}-{los['max']})."))
        else:
            out.append(RuleResult("R08", "Length of stay within STG", "fail", f"{n} days outside allowed {los['min']}-{los['max']}."))
    else:
        out.append(RuleResult("R08", "Length of stay within STG", "na", "Admission/discharge dates not supplied."))

    # R09 stage of care / episode timeline
    img_d, proc_d = _d(meta.imaging_date), _d(meta.procedure_date)
    problems = []
    if meta.stage_of_care == "claim_settlement":
        need_proc = pkg.get("requires_procedure_date", True)
        if not (adm and dis and (proc_d or not need_proc)):
            problems.append("claim settlement needs admission, discharge" + (" and procedure" if need_proc else "") + " dates")
    if img_d and proc_d and img_d > proc_d:
        problems.append("diagnostic imaging is dated AFTER the procedure")
    if proc_d and adm and dis and not (adm <= proc_d <= dis):
        problems.append("procedure falls outside the admission window")
    if img_d and adm and (adm - img_d).days > 30:
        problems.append("imaging is >30 days before admission (stale evidence)")
    has_dates = any([adm, dis, proc_d, img_d])
    out.append(RuleResult("R09", f"Stage of care & episode timeline ({meta.stage_of_care})",
                          "fail" if problems else ("pass" if has_dates else "warn"),
                          "; ".join(problems) if problems else ("Timeline is coherent." if has_dates else "No dates supplied to verify timeline.")))

    # R10 billed procedure documented in report
    rp = {p.name for p in report.procedures}
    reqp = set(pkg.get("required_procedures", []))
    if reqp:
        out.append(RuleResult("R10", "Billed procedure documented in report",
                              "pass" if rp & reqp else ("warn" if meta.stage_of_care == "pre_authorization" else "fail"),
                              f"Report procedures: {', '.join(sorted(rp)) or 'none'}; expected one of {', '.join(sorted(reqp))}."))
    rec = set(pkg.get("recommended_report_terms", []))
    if rec:
        out.append(RuleResult("R11", "Confirmatory work-up documented", "pass" if rp & rec else "warn",
                              f"Recommended: {', '.join(sorted(rec))}; found: {', '.join(sorted(rp & rec)) or 'none'}."))
    return out, pkg
