import os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claimlens import pipeline, scenarios
from claimlens.dedupe import Registry
from claimlens.nlp.entities import RuleBasedExtractor
from claimlens.preprocessing import assess_quality
import numpy as np


def run(name, tmp, reg=None, commit=False):
    meta, imgs, rep, _ = scenarios.build(name)
    return pipeline.analyze_claim(meta, imgs, rep, out_root=tmp, registry=reg or Registry(Path(tmp) / "r.json"), commit=commit)


def codes(r):
    return {f["code"] for f in r["flags"]}


class TestNLP(unittest.TestCase):
    def test_negation_and_laterality(self):
        e = RuleBasedExtractor().extract("No pleural effusion. Left femur comminuted fracture. Right lung clear, no infiltrates.")
        by = {(f.label, f.negated) for f in e.findings}
        self.assertIn(("effusion", True), by)
        self.assertIn(("fracture", False), by)
        self.assertIn(("infiltrate", True), by)
        fr = [f for f in e.findings if f.label == "fracture"][0]
        self.assertEqual((fr.laterality, fr.severity), ("left", "severe"))

    def test_conservative_days(self):
        e = RuleBasedExtractor().extract("Conservative management with physiotherapy for 6 weeks.")
        self.assertEqual(e.conservative_days, 42)


class TestQuality(unittest.TestCase):
    def test_flags_low_contrast_and_resolution(self):
        q = assess_quality(np.full((200, 200), 120, np.uint8))
        self.assertIn("LOW_CONTRAST", q["flags"]); self.assertIn("LOW_RESOLUTION", q["flags"])


class TestScenarios(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_consistent_claim_is_low_risk(self):
        r = run("ortho_consistent", self.tmp)
        self.assertEqual(r["recommendation"]["band"], "low")
        self.assertEqual(r["flags"], [])

    def test_exaggeration(self):
        r = run("ortho_exaggerated", self.tmp)
        self.assertIn("EXAGGERATION", codes(r))

    def test_unsupported(self):
        self.assertIn("UNSUPPORTED_FINDING", codes(run("ortho_no_fracture", self.tmp)))

    def test_contradiction_escalates(self):
        r = run("chest_contradiction", self.tmp)
        self.assertIn("CONTRADICTION", codes(r)); self.assertEqual(r["recommendation"]["band"], "high")

    def test_wrong_laterality(self):
        self.assertIn("LATERALITY_MISMATCH", codes(run("chest_wrong_side", self.tmp)))

    def test_scanned_report_ocr_path(self):
        r = run("pneumonia_scanned_report", self.tmp)
        self.assertEqual(r["report"]["source"], "ocr-image"); self.assertEqual(r["recommendation"]["band"], "low")

    def test_stg_conservative_timeline(self):
        r = run("tkr_insufficient_conservative", self.tmp)
        self.assertTrue(any(c["rule_id"] == "R07" and c["status"] == "fail" for c in r["stg_checks"]))

    def test_recycled_image_across_claims(self):
        reg = Registry(Path(self.tmp) / "reg.json")
        run("ortho_consistent", self.tmp, reg, commit=True)
        meta, imgs, rep, _ = scenarios.build("ortho_consistent")
        meta.claim_id = "SOME-OTHER-CLAIM"
        r = pipeline.analyze_claim(meta, imgs, rep, out_root=self.tmp, registry=reg, commit=False)
        self.assertIn("RECYCLED_IMAGE", codes(r))

    def test_outputs_written(self):
        r = run("chest_wrong_side", self.tmp)
        d = Path(self.tmp) / r["claim"]["claim_id"]
        for f in ("result.json", "img1_explain.png", "img1_enhanced.png", "img1_original.png"):
            self.assertTrue((d / f).exists(), f)


class TestWeb(unittest.TestCase):
    def test_pages_and_upload(self):
        os.environ["CLAIMLENS_OUTPUT"] = tempfile.mkdtemp()
        os.environ["CLAIMLENS_UPLOADS"] = tempfile.mkdtemp()
        os.environ["CLAIMLENS_REGISTRY"] = str(Path(tempfile.mkdtemp()) / "reg.json")
        for m in [k for k in sys.modules if k.startswith("claimlens") or k == "app"]:
            del sys.modules[m]
        import app as webapp
        c = webapp.app.test_client()
        self.assertEqual(c.get("/").status_code, 200)
        sd = Path(__file__).resolve().parent.parent / "sample_data"
        data = {"images": (open(sd / "chest_effusion_patient_right.png", "rb"), "x.png"),
                "report_text": "Left pleural effusion. Intercostal drainage on the left.",
                "package_code": "PU-EFFUSION-DRAIN", "modality": "xray", "body_part": "chest",
                "billed_laterality": "left", "claim_id": "WEBTEST"}
        r = c.post("/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 302)
        page = c.get(r.headers["Location"])
        self.assertEqual(page.status_code, 200); self.assertIn(b"Wrong side", page.data)
        self.assertEqual(c.post("/api/analyze", data={"package_code": "x"}).status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
