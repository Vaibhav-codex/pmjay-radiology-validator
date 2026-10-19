"""Flask web app: upload claim images + report, get an adjudication-support report.

    python app.py            ->  http://127.0.0.1:5000
"""
import json
import re
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for

from claimlens import config, pipeline, scenarios
from claimlens.schemas import ClaimMeta
from claimlens.stg.validator import load_rules

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
ALLOWED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm", ".webp"}
ALLOWED_REP = {".txt", ".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80] or "file"


def _int(v):
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


def _meta_from_form(f):
    cid = _safe(f.get("claim_id") or f"CLM-{uuid.uuid4().hex[:8].upper()}")
    return ClaimMeta(
        claim_id=cid, package_code=f.get("package_code", ""), stage_of_care=f.get("stage_of_care", "claim_settlement"),
        modality=f.get("modality", "auto"), body_part=f.get("body_part", "auto"),
        billed_laterality=f.get("billed_laterality", "unknown"), image_marker_laterality=f.get("image_marker_laterality", "unknown"),
        admission_date=f.get("admission_date") or None, discharge_date=f.get("discharge_date") or None,
        procedure_date=f.get("procedure_date") or None, imaging_date=f.get("imaging_date") or None,
        conservative_days=_int(f.get("conservative_days")), patient_age=_int(f.get("patient_age")))


def _run_upload():
    meta = _meta_from_form(request.form)
    up_dir = config.UPLOAD_DIR / meta.claim_id
    up_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for f in request.files.getlist("images"):
        if f and f.filename and Path(f.filename).suffix.lower() in ALLOWED_IMG:
            p = up_dir / _safe(f.filename)
            f.save(p)
            images.append(p)
    report_path = None
    rf = request.files.get("report_file")
    if rf and rf.filename and Path(rf.filename).suffix.lower() in ALLOWED_REP:
        report_path = up_dir / _safe(rf.filename)
        rf.save(report_path)
    report_text = request.form.get("report_text", "")
    if not images:
        raise ValueError("Please upload at least one radiology image (JPG / PNG / DICOM).")
    if not report_path and not report_text.strip():
        raise ValueError("Please upload a report file or paste the report text.")
    return pipeline.analyze_claim(meta, images, report_path, report_text)


@app.get("/")
def index():
    rules = load_rules()
    return render_template("index.html", packages=rules["packages"], scenarios=scenarios.load_scenarios(), error=request.args.get("error"))


@app.post("/analyze")
def analyze():
    try:
        res = _run_upload()
    except Exception as e:      # surface friendly errors in the form
        return redirect(url_for("index", error=str(e)))
    return redirect(url_for("claim", claim_id=res["claim"]["claim_id"]))


@app.post("/api/analyze")
def api_analyze():
    try:
        res = _run_upload()
    except Exception as e:
        return jsonify(error=str(e)), 400
    res.pop("report", None) if request.args.get("brief") else None
    return jsonify(res)


@app.get("/demo/<name>")
def demo(name):
    if name not in scenarios.load_scenarios():
        abort(404)
    meta, images, report, _ = scenarios.build(name)
    res = pipeline.analyze_claim(meta, images, report, commit=False)
    return redirect(url_for("claim", claim_id=res["claim"]["claim_id"]))


@app.get("/claim/<claim_id>")
def claim(claim_id):
    p = config.OUTPUT_DIR / _safe(claim_id) / "result.json"
    if not p.exists():
        abort(404)
    return render_template("result.html", r=json.loads(p.read_text()))


@app.get("/claim/<claim_id>.json")
def claim_json(claim_id):
    p = config.OUTPUT_DIR / _safe(claim_id) / "result.json"
    if not p.exists():
        abort(404)
    return app.response_class(p.read_text(), mimetype="application/json")


@app.get("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(config.OUTPUT_DIR, filename)


@app.template_filter("pct")
def pct(v):
    return f"{float(v) * 100:.0f}%"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
