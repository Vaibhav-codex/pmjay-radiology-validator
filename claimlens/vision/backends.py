"""Backend registry. Deep-learning backends are used automatically when their
libraries are installed; otherwise the heuristic backend runs."""
import cv2
import numpy as np
from ..schemas import Finding
from .base import VisionBackend, VisionResult
from .heuristic import HeuristicBackend


class TorchXRayVisionBackend(VisionBackend):
    """Chest X-ray multi-label classifier (DenseNet121, TorchXRayVision) + Grad-CAM.
    Requires: torch, torchvision, torchxrayvision.   (Not exercised in the offline build sandbox.)"""
    name = "torchxrayvision-densenet121"
    MAP = {"Pneumonia": "infiltrate", "Infiltration": "infiltrate", "Consolidation": "infiltrate",
           "Lung Opacity": "infiltrate", "Effusion": "effusion", "Mass": "mass", "Nodule": "mass",
           "Fracture": "fracture"}
    detectable = set(MAP.values())

    def __init__(self):
        import torch, torchvision, torchxrayvision as xrv
        self.torch, self.xrv = torch, xrv
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
        self.tf = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
        self._acts = self._grads = None
        layer = self.model.features
        layer.register_forward_hook(lambda m, i, o: setattr(self, "_acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "_grads", go[0]))

    def analyze(self, gray, modality, body_part):
        torch, xrv = self.torch, self.xrv
        img = xrv.datasets.normalize(gray, 255)[None]
        img = self.tf(img)
        x = torch.from_numpy(img)[None].float().requires_grad_(True)
        out = self.model(x)[0]
        paths = self.model.pathologies
        best, heat = {}, np.zeros((224, 224), np.float32)
        for name, label in self.MAP.items():
            if name in paths:
                p = float(out[paths.index(name)])
                if label not in best or p > best[label][0]:
                    best[label] = (p, name)
        findings = []
        for label, (p, name) in best.items():
            idx = paths.index(name)
            self.model.zero_grad()
            out = self.model(x)[0]
            out[idx].backward(retain_graph=True)
            w = self._grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * self._acts).sum(1))[0].detach().cpu().numpy()
            cam = cv2.resize(cam, (224, 224))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
            ys, xs = np.nonzero(cam > 0.6)
            bbox = [xs.min() / 224, ys.min() / 224, (xs.max() + 1) / 224, (ys.max() + 1) / 224] if len(ys) else None
            if p >= 0.35:
                heat = np.maximum(heat, cam * p)
            findings.append(Finding(label, round(p, 2), bbox, "unknown", "unknown",
                                    f"{name} probability {p:.2f} (TorchXRayVision)"))
        h, w = gray.shape
        return VisionResult(findings, cv2.resize(heat, (w, h)), self.detectable,
                            ["Grad-CAM from DenseNet121 features"])


class YoloLesionBackend(VisionBackend):
    """Placeholder for a fine-tuned YOLOv8/11 focal-lesion detector (fractures, nodules, masses).
    Provide weights via CLAIMLENS_YOLO_WEIGHTS."""
    name = "yolo-lesion"
    detectable = {"fracture", "mass"}

    def __init__(self, weights):
        from ultralytics import YOLO
        self.model = YOLO(weights)

    def analyze(self, gray, modality, body_part):
        res = self.model.predict(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), verbose=False)[0]
        h, w = gray.shape
        findings = []
        for b in res.boxes:
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            findings.append(Finding(res.names[int(b.cls)], round(float(b.conf), 2), [x0 / w, y0 / h, x1 / w, y1 / h]))
        return VisionResult(findings, None, self.detectable)


_cache = {}


def get_backend(modality: str, body_part: str) -> VisionBackend:
    """Pick the best available backend for this image."""
    if modality in ("xray", "auto") and body_part in ("chest", "auto"):
        if "xrv" not in _cache:
            try:
                _cache["xrv"] = TorchXRayVisionBackend()
            except Exception:
                _cache["xrv"] = None
        if _cache["xrv"] is not None and body_part == "chest":
            return _cache["xrv"]
    return _cache.setdefault("heur", HeuristicBackend())
