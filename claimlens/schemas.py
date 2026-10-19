from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ClaimMeta:
    claim_id: str = "CLAIM-0001"
    package_code: str = ""
    stage_of_care: str = "claim_settlement"      # pre_authorization | during_treatment | claim_settlement
    modality: str = "auto"                        # auto | xray | ct | mri | ultrasound
    body_part: str = "auto"                       # auto | chest | long_bone | knee | hand | abdomen | other
    billed_laterality: str = "unknown"            # left | right | bilateral | unknown
    image_marker_laterality: str = "unknown"      # L/R marker visible on the film (if any)
    admission_date: Optional[str] = None          # ISO yyyy-mm-dd
    discharge_date: Optional[str] = None
    procedure_date: Optional[str] = None
    imaging_date: Optional[str] = None
    conservative_days: Optional[int] = None
    patient_age: Optional[int] = None


@dataclass
class Finding:
    label: str
    confidence: float
    bbox: Optional[list] = None                   # normalised [x0, y0, x1, y1]
    laterality: str = "unknown"
    severity: str = "unknown"
    description: str = ""
    metrics: dict = field(default_factory=dict)
    image_id: str = ""


@dataclass
class ImageAnalysis:
    image_id: str
    filename: str
    modality: str
    body_part: str
    backend: str
    quality: dict
    findings: list
    artifacts: dict = field(default_factory=dict)  # name -> relative path


@dataclass
class ReportFinding:
    label: str
    text: str
    negated: bool
    severity: str
    laterality: str
    site: str
    sentence: str
    start: int = 0                                 # char span inside full report text
    end: int = 0


@dataclass
class ReportProcedure:
    name: str
    text: str
    laterality: str = "unknown"
    start: int = 0
    end: int = 0


@dataclass
class ReportEntities:
    text: str
    findings: list
    procedures: list
    sites: list
    conservative_days: Optional[int]
    ocr_confidence: Optional[float] = None
    extractor: str = "rule-based"


@dataclass
class Flag:
    code: str
    severity: str                                  # high | medium | low | info
    title: str
    rationale: str
    evidence: dict = field(default_factory=dict)


@dataclass
class RuleResult:
    rule_id: str
    title: str
    status: str                                    # pass | fail | warn | na
    detail: str


def to_dict(obj):
    return asdict(obj)
