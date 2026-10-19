"""Command-line interface.

Examples
    python run_cli.py --demo chest_wrong_side
    python run_cli.py --images xray1.jpg xray2.jpg --report report.pdf --package OR-FRACT-ORIF \
        --modality xray --body-part long_bone --billed-laterality left --claim-id CLM-1001
"""
import argparse
import json
import sys

from claimlens import pipeline, scenarios
from claimlens.schemas import ClaimMeta


def main():
    ap = argparse.ArgumentParser(description="PM-JAY radiology/report cross-verification")
    ap.add_argument("--demo", help="run a bundled scenario (see sample_data/scenarios.yaml)")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--report", help="report file (.txt/.pdf/.png/.jpg)")
    ap.add_argument("--report-text")
    ap.add_argument("--claim-id", default="CLI-0001")
    ap.add_argument("--package", default="")
    ap.add_argument("--stage", default="claim_settlement", choices=["pre_authorization", "during_treatment", "claim_settlement"])
    ap.add_argument("--modality", default="auto", choices=["auto", "xray", "ct", "mri", "ultrasound"])
    ap.add_argument("--body-part", default="auto")
    ap.add_argument("--billed-laterality", default="unknown")
    ap.add_argument("--admission"); ap.add_argument("--discharge"); ap.add_argument("--procedure"); ap.add_argument("--imaging")
    ap.add_argument("--conservative-days", type=int)
    ap.add_argument("--json", action="store_true", help="print full JSON")
    a = ap.parse_args()

    if a.demo:
        meta, images, report, title = scenarios.build(a.demo)
        res = pipeline.analyze_claim(meta, images, report, commit=False)
    else:
        if not a.images or not (a.report or a.report_text):
            ap.error("--images and --report/--report-text are required (or use --demo)")
        meta = ClaimMeta(claim_id=a.claim_id, package_code=a.package, stage_of_care=a.stage, modality=a.modality,
                         body_part=a.body_part, billed_laterality=a.billed_laterality, admission_date=a.admission,
                         discharge_date=a.discharge, procedure_date=a.procedure, imaging_date=a.imaging,
                         conservative_days=a.conservative_days)
        res = pipeline.analyze_claim(meta, a.images, a.report, a.report_text)

    if a.json:
        print(json.dumps(res, indent=2, default=str)); return
    rec = res["recommendation"]
    print(f"\nClaim {res['claim']['claim_id']}  |  package {res['package'].get('name', res['claim']['package_code'])}")
    print(f"Risk score: {res['risk_score']}/100  ->  {rec['action']}  (alignment {res['alignment_score']})")
    print(rec["summary"], "\n")
    print("Image findings:")
    for k, v in res["image_findings_aggregated"].items():
        print(f"  - {v['display']}: conf {v['confidence']}, {v['severity']}, side={v['laterality']}  {v['description']}")
    print("\nDiscrepancy flags:")
    for f in res["flags"]:
        print(f"  [{f['severity'].upper():6}] {f['title']}")
    print("\nSTG / package checks:")
    for c in res["stg_checks"]:
        print(f"  [{c['status'].upper():4}] {c['rule_id']} {c['title']} - {c['detail']}")
    print(f"\nArtifacts (heatmaps, JSON): outputs/{res['claim']['claim_id']}/")
    print(rec["disclaimer"])


if __name__ == "__main__":
    sys.exit(main())
