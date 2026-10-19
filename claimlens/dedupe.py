"""Repeated / multiple-submission detection using perceptual hashes (dHash)."""
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from . import config


def dhash(gray: np.ndarray, size: int = 8) -> int:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return int("".join("1" if b else "0" for b in diff.flatten()), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def text_hash(text: str) -> str:
    norm = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(norm.encode()).hexdigest()


class Registry:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.REGISTRY_PATH)
        self.data = {"images": [], "reports": []}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                pass

    def check_image(self, claim_id: str, filename: str, gray: np.ndarray):
        h = dhash(gray)
        hits = []
        for rec in self.data["images"]:
            d = hamming(h, int(rec["hash"]))
            if d <= config.T["dup_hamming"]:
                hits.append(dict(claim_id=rec["claim_id"], filename=rec["filename"], distance=d, ts=rec["ts"]))
        return h, hits

    def check_report(self, claim_id: str, text: str):
        h = text_hash(text) if text.strip() else None
        hits = [r for r in self.data["reports"] if h and r["hash"] == h]
        return h, hits

    def commit(self, claim_id, images, report_hash):
        now = int(time.time())
        for fn, h in images:
            self.data["images"].append(dict(claim_id=claim_id, filename=fn, hash=str(h), ts=now))
        if report_hash:
            self.data["reports"].append(dict(claim_id=claim_id, hash=report_hash, ts=now))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))
