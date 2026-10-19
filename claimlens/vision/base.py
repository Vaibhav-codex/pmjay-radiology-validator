from abc import ABC, abstractmethod
import numpy as np


class VisionResult:
    def __init__(self, findings, heatmap=None, detectable=None, notes=None):
        self.findings = findings            # list[Finding]
        self.heatmap = heatmap              # optional float array 0..1 (same shape as input) e.g. Grad-CAM
        self.detectable = set(detectable or [])   # labels this backend can actually detect
        self.notes = notes or []


class VisionBackend(ABC):
    name = "base"

    @abstractmethod
    def analyze(self, gray: np.ndarray, modality: str, body_part: str) -> VisionResult:
        ...
