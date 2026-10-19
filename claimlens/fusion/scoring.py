from .. import config


def risk_score(flags, rules):
    pts = 0
    for f in flags:
        pts += config.RISK_WEIGHTS.get(f.code, 5)
    for r in rules:
        if r.status == "fail":
            pts += config.STG_FAIL_POINTS
        elif r.status == "warn":
            pts += config.STG_WARN_POINTS
    return min(100, pts)


def recommendation(score, flags, rules):
    high = [f for f in flags if f.severity == "high"]
    fails = [r for r in rules if r.status == "fail"]
    if score >= 60 or len(high) >= 2:
        band, action = "high", "Escalate to medical audit"
        text = ("Multiple strong inconsistencies between image evidence, report and package. "
                "Recommend detailed clinical review and a query to the hospital before any payout decision.")
    elif score >= 25 or high or fails:
        band, action = "medium", "Refer for detailed review"
        text = "Some findings need adjudicator attention. Review the flagged items and highlighted image regions; consider querying the hospital."
    else:
        band, action = "low", "Evidence consistent"
        text = "Image findings, report and STG checks are broadly aligned. Suitable for routine adjudication."
    return dict(band=band, action=action, summary=text,
                disclaimer="Decision support only - the final adjudication decision rests with the human reviewer.")
