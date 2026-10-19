# ClaimLens — Radiology ↔ Report Cross-Verification for AB PM-JAY Claims

Assistive AI for NHA claim adjudicators (Problem Statement 2). It reads the radiology images and the hospital report
submitted with a claim, checks whether they agree, validates the billed package against Standard Treatment Guidelines,
and shows its evidence (heatmaps, boxes, plain-language reasons). **It supports the human reviewer; it never decides.**

## What you give it → what you get

| Input (web form / CLI / API) | Output |
|---|---|
| 1..n images: JPG, PNG, DICOM (X-ray, CT, MRI, ultrasound) | Per-image **quality report**, enhanced image, **heatmap + bounding boxes** |
| Report: TXT, text PDF, or scanned/photographed PDF/image (OCR) | Extracted entities: findings (with negation), severity, side, procedures, conservative-care days |
| Claim data: package, stage of care, billed side, imaging/admission/procedure/discharge dates | **Discrepancy flags** (omission, contradiction, exaggeration, unsupported, wrong side, recycled image…) |
| | **STG checklist** (diagnosis, imaging support, severity-vs-tier, modality, laterality, conservative timeline, LOS, episode timeline, procedure) |
| | **Risk score 0-100**, recommendation band, `result.json` |

## Workflow (matches the problem statement)

```
Claim ─┬─ images ──► quality check + enhance ─► vision backend ─► findings + heatmap ─┐
       │                                                                              ├─► cross-modal
       └─ report ──► OCR / PDF text ─► clinical entity extraction (NegEx-style) ──────┘   discrepancy engine
                                                                                              │
                         adjudicator dashboard ◄── risk score ◄── STG / package rule engine ◄─┘
```

## File structure

```
pmjay-radiology-validator/
├── app.py                      Flask web app (form, results dashboard, JSON API)
├── run_cli.py                  Command-line runner
├── requirements.txt
├── config/
│   ├── stg_rules.yaml          Package + STG rules (ILLUSTRATIVE – replace with official NHA data)
│   └── clinical_lexicon.yaml   Findings / severity / negation / laterality / procedure vocabulary
├── claimlens/
│   ├── config.py               Paths, thresholds, risk weights
│   ├── schemas.py              Dataclasses (ClaimMeta, Finding, Flag, RuleResult…)
│   ├── ontology.py             Canonical findings → SNOMED-CT / ICD-10 (indicative)
│   ├── ingestion.py            DICOM/JPG/PNG + TXT/PDF/scan loaders
│   ├── preprocessing.py        Border crop, quality QC, denoise + CLAHE
│   ├── dedupe.py               Perceptual-hash registry → repeated / recycled submissions
│   ├── pipeline.py             End-to-end orchestration
│   ├── scenarios.py            Bundled demo scenarios
│   ├── vision/
│   │   ├── base.py             Backend interface
│   │   ├── heuristic.py        No-weights fallback (chest, bone, ultrasound/CT/MRI)
│   │   ├── backends.py         TorchXRayVision+Grad-CAM, YOLO plug-ins, backend selector
│   │   └── explain.py          Heatmap + bounding-box rendering
│   ├── nlp/
│   │   ├── ocr.py              Tesseract OCR with deskew / adaptive threshold; PDF handling
│   │   ├── entities.py         Rule-based extractor + report highlighting
│   │   └── transformer_ner.py  Hook for ClinicalBERT / BioLinkBERT
│   ├── fusion/
│   │   ├── discrepancy.py      Omission / contradiction / exaggeration / laterality / site logic
│   │   └── scoring.py          Risk score + recommendation
│   └── stg/validator.py        Deterministic STG rule engine
├── templates/  static/         Intake form + adjudicator dashboard
├── scripts/make_sample_data.py Generates synthetic phantoms, reports, scenarios
├── sample_data/                Demo images / reports / scenarios.yaml
├── tests/test_pipeline.py      13 tests (NLP, QC, every scenario, dedupe, web upload)
├── outputs/                    Per-claim results (JSON + images)
└── data/registry.json          Hash registry for duplicate detection (created at runtime)
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install tesseract-ocr          # macOS: brew install tesseract   (needed for scanned reports)

python app.py                            # open http://127.0.0.1:5000
```

* **Web:** upload images + report (file or pasted text), choose the billed package, submit. Or click a bundled scenario.
* **CLI:**
  ```bash
  python run_cli.py --demo chest_wrong_side
  python run_cli.py --images xr1.jpg xr2.jpg --report discharge.pdf --package OR-FRACT-ORIF \
      --modality xray --body-part long_bone --billed-laterality left --admission 2026-09-01 \
      --procedure 2026-09-02 --discharge 2026-09-07 --claim-id CLM-1001
  ```
* **API:** `curl -F images=@xr.png -F report_text="…" -F package_code=PU-EFFUSION-DRAIN -F body_part=chest http://127.0.0.1:5000/api/analyze`
* **Tests:** `python -m unittest discover -s tests -v`
* Regenerate sample data: `python scripts/make_sample_data.py`

## Bundled scenarios (synthetic, no patient data)

| Scenario | Expected result |
|---|---|
| `ortho_consistent` | Low risk, no flags |
| `ortho_exaggerated` | EXAGGERATION: "severe comminuted" reported, hairline on image; STG severity check fails |
| `ortho_no_fracture` | UNSUPPORTED_FINDING: fracture claimed, image intact |
| `chest_contradiction` | CONTRADICTION: report says clear lungs, image shows effusion → escalate |
| `chest_wrong_side` | LATERALITY_MISMATCH: left drained/billed, effusion on the right |
| `pneumonia_scanned_report` | Consistent; report read through OCR |
| `tkr_insufficient_conservative` | STG R07 fail: 21 of required 90 conservative days |

## Plugging in real models

* **Chest X-ray:** `pip install torch torchvision torchxrayvision` → `get_backend()` automatically uses
  `TorchXRayVisionBackend` (DenseNet121 multi-label + Grad-CAM) for chest films.
* **Focal lesions (fracture, nodule, mass):** train YOLOv8/11 → implement/enable `YoloLesionBackend`.
* **CT / MRI volumes:** add a MONAI / Swin-UNETR backend implementing `VisionBackend.analyze` (return `Finding`s + optional heatmap).
* **Segmentation metrics (tumour volume, % infiltration):** put nnU-Net outputs into `Finding.metrics`.
* **Report NLP:** fine-tune ClinicalBERT/BioLinkBERT, complete `nlp/transformer_ner.py`; contrastive image-text scoring (MedCLIP/BioViL) can feed `alignment_score`.
* **Real STGs:** edit `config/stg_rules.yaml` with the official NHA package master and STG thresholds.
* **Calibrate** thresholds in `claimlens/config.py` (`T`, `RISK_WEIGHTS`) on labelled adjudication outcomes.

## Honest limitations

* The built-in image analysis is **classical-CV, demonstration-grade** — tuned on synthetic phantoms, not validated on real
  radiographs. Do not use its findings for real adjudication until a trained, validated backend is plugged in.
* The TorchXRayVision and YOLO backends are written but could **not be executed in the offline build environment**; test them after installing the libraries.
* Package codes, thresholds and rules in `stg_rules.yaml` are **placeholders**, not the official PM-JAY STGs. SNOMED/ICD mappings are indicative.
* Image laterality can only be inferred automatically for paired organs (chest); for limbs, supply the film marker or rely on report/billing consistency.
* The extractor is lexicon-based; unusual phrasing, other languages (Hindi/regional) or poor OCR will reduce recall.
* Perceptual-hash dedupe catches re-submitted or re-photographed copies of the same film, not different films of the same patient.
