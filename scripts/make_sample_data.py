"""Generate SYNTHETIC radiographs + reports + scenarios for demos/tests.
These are cartoon-level phantoms (no patient data) - useful only to exercise the pipeline."""
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "sample_data"
rng = np.random.default_rng(7)
S = 768


def finish(img, noise=4.0):
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.2)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def chest(effusion=None, infiltrate=None, mass=None):
    """effusion / infiltrate: 'viewer_left' (= patient RIGHT) or 'viewer_right' (= patient LEFT)."""
    img = np.full((S, S), 150, np.float32)
    yy, xx = np.mgrid[:S, :S]
    cv2.ellipse(img, (S // 2, int(S * .5)), (int(S * .08), int(S * .4)), 0, 0, 360, 185, -1)   # mediastinum
    lungs = {}
    for side, cx in (("viewer_left", int(S * .31)), ("viewer_right", int(S * .69))):
        cv2.ellipse(img, (cx, int(S * .48)), (int(S * .17), int(S * .33)), 0, 0, 360, 55, -1)
        lungs[side] = cx
    for side, cx in lungs.items():
        if effusion == side:
            m = (((xx - cx) / (S * .17)) ** 2 + ((yy - S * .48) / (S * .33)) ** 2 <= 1) & (yy > S * .48 + S * .33 * 0.30)
            img[m] = 170
        if infiltrate == side:
            m = (((xx - cx) / (S * .17)) ** 2 + ((yy - S * .42) / (S * .33)) ** 2 <= 1) & (yy > S * .30) & (yy < S * .62)
            img[m] = 120 + 25 * rng.random(m.sum())
        if mass == side:
            cv2.circle(img, (cx - 10, int(S * .35)), int(S * .05), 175, -1)
    return finish(img)


def bone(lines=0, fragments=False):
    img = np.full((S, S), 45, np.float32)
    for x in range(S):  # tapering shaft with slight curve
        cx = S // 2 + int(25 * np.sin(x / 250))
        pass
    cv2.rectangle(img, (S // 2 - 70, 40), (S // 2 + 70, S - 40), 190, -1)
    cv2.rectangle(img, (S // 2 - 45, 40), (S // 2 + 45, S - 40), 140, -1)               # medullary canal
    img += rng.normal(0, 6, img.shape)
    pts = [((S // 2 - 90, 330), (S // 2 + 90, 400)), ((S // 2 - 90, 470), (S // 2 + 90, 430)), ((S // 2 - 90, 380), (S // 2 + 90, 360))]
    for k in range(lines):
        a, b = pts[k]
        cv2.line(img, a, b, 25, 5)
    if fragments:
        cv2.circle(img, (S // 2 + 30, 410), 14, 205, -1)
    return finish(img, 3)


def ultrasound(fluid=True):
    img = np.zeros((S, S), np.float32)
    cv2.ellipse(img, (S // 2, 0), (330, 700), 0, 20, 160, 1, -1)
    fan = np.zeros((S, S), np.uint8)
    cv2.ellipse(fan, (S // 2, 0), (330, 700), 0, 20, 160, 255, -1)
    tex = 95 + rng.normal(0, 22, (S, S))
    tex = cv2.GaussianBlur(tex.astype(np.float32), (0, 0), 1.5)
    img = np.where(fan > 0, tex, 0)
    if fluid:
        cv2.ellipse(img, (S // 2 + 30, 360), (110, 80), 0, 0, 360, 4, -1)
    return finish(img, 2)


def scanned_report(text, path):
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:
        font = ImageFont.load_default()
    im = Image.new("L", (1400, 900), 235)
    d = ImageDraw.Draw(im)
    y = 40
    for line in text.splitlines():
        d.text((50, y), line, fill=30, font=font)
        y += 40
    im = im.rotate(1.2, expand=False, fillcolor=235)
    arr = np.array(im).astype(np.float32) + rng.normal(0, 6, (900, 1400))
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def save(name, arr):
    Image.fromarray(arr).save(OUT / name)


def main():
    OUT.mkdir(exist_ok=True)
    save("femur_comminuted.png", bone(3, True))
    save("femur_hairline.png", bone(1))
    save("femur_normal.png", bone(0))
    save("chest_normal.png", chest())
    save("chest_effusion_patient_right.png", chest(effusion="viewer_left"))
    save("chest_pneumonia_patient_left.png", chest(infiltrate="viewer_right"))
    save("abdomen_us_fluid.png", ultrasound(True))

    reports = {
        "report_femur_comminuted.txt": """DISCHARGE SUMMARY
X-ray left femur: comminuted fracture of the mid-shaft with multiple fragments.
Patient underwent open reduction and internal fixation with intramedullary nailing of the left femur.
Post-operative course uneventful.""",
        "report_femur_exaggerated.txt": """DISCHARGE SUMMARY
X-ray left femur shows a severe comminuted displaced fracture of the shaft with multiple fragments.
Patient underwent open reduction and internal fixation with plating of the left femur.""",
        "report_chest_normal_but_effusion.txt": """CHEST X-RAY REPORT
Both lungs are clear. No pleural effusion. No infiltrates. Cardiac size normal.
Patient treated with IV antibiotics for pneumonia.""",
        "report_effusion_wrong_side.txt": """CHEST X-RAY REPORT
Moderate left pleural effusion with blunting of the left costophrenic angle.
Intercostal drainage performed on the left side.""",
        "report_pneumonia_ok.txt": """CHEST X-RAY REPORT
Left lower zone consolidation with increased opacity in the left lung, consistent with pneumonia.
Right lung is clear. Managed with IV antibiotics and oxygen therapy.""",
        "report_knee_oa.txt": """ORTHOPAEDIC SUMMARY
Weight-bearing knee X-ray of the right knee shows moderate osteoarthritis with joint space narrowing and osteophytes.
Conservative management with physiotherapy and NSAIDs for 3 weeks. Right total knee replacement performed.""",
    }
    for n, t in reports.items():
        (OUT / n).write_text(t)
    scanned_report(reports["report_pneumonia_ok.txt"], OUT / "report_pneumonia_ok_scanned.png")

    scenarios = {
        "ortho_consistent": dict(title="Orthopaedic ORIF - everything consistent", package_code="OR-FRACT-ORIF", modality="xray",
            body_part="long_bone", billed_laterality="left", images=["femur_comminuted.png"], report="report_femur_comminuted.txt",
            admission_date="2026-09-01", procedure_date="2026-09-02", discharge_date="2026-09-07", imaging_date="2026-09-01"),
        "ortho_exaggerated": dict(title="ORIF billed on a hairline fracture (severity exaggerated)", package_code="OR-FRACT-ORIF", modality="xray",
            body_part="long_bone", billed_laterality="left", images=["femur_hairline.png"], report="report_femur_exaggerated.txt",
            admission_date="2026-09-01", procedure_date="2026-09-02", discharge_date="2026-09-06", imaging_date="2026-09-01"),
        "ortho_no_fracture": dict(title="Fracture claimed, image intact (unsupported)", package_code="OR-FRACT-ORIF", modality="xray",
            body_part="long_bone", billed_laterality="left", images=["femur_normal.png"], report="report_femur_exaggerated.txt",
            admission_date="2026-09-01", procedure_date="2026-09-02", discharge_date="2026-09-06", imaging_date="2026-09-01"),
        "chest_contradiction": dict(title="Report says clear lungs, image shows effusion", package_code="PU-PNEUMONIA-MED", modality="xray",
            body_part="chest", images=["chest_effusion_patient_right.png"], report="report_chest_normal_but_effusion.txt",
            admission_date="2026-09-10", discharge_date="2026-09-14", imaging_date="2026-09-10"),
        "chest_wrong_side": dict(title="Left effusion drained - image shows RIGHT effusion", package_code="PU-EFFUSION-DRAIN", modality="xray",
            body_part="chest", billed_laterality="left", images=["chest_effusion_patient_right.png"], report="report_effusion_wrong_side.txt",
            admission_date="2026-09-10", procedure_date="2026-09-11", discharge_date="2026-09-16", imaging_date="2026-09-10"),
        "pneumonia_scanned_report": dict(title="Pneumonia, scanned (photographed) report - consistent", package_code="PU-PNEUMONIA-MED", modality="xray",
            body_part="chest", images=["chest_pneumonia_patient_left.png"], report="report_pneumonia_ok_scanned.png",
            admission_date="2026-09-10", discharge_date="2026-09-15", imaging_date="2026-09-10"),
        "tkr_insufficient_conservative": dict(title="TKR without the required conservative-management period", package_code="OR-KNEE-TKR", modality="xray",
            body_part="knee", billed_laterality="right", images=["femur_normal.png"], report="report_knee_oa.txt",
            admission_date="2026-09-01", procedure_date="2026-09-02", discharge_date="2026-09-08", imaging_date="2026-09-01"),
    }
    (OUT / "scenarios.yaml").write_text(yaml.safe_dump(scenarios, sort_keys=False))
    print("wrote", len(list(OUT.iterdir())), "files to", OUT)


if __name__ == "__main__":
    main()
