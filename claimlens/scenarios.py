"""Load demo scenarios from sample_data/scenarios.yaml."""
import yaml
from . import config
from .schemas import ClaimMeta


def load_scenarios():
    p = config.SAMPLE_DIR / "scenarios.yaml"
    return yaml.safe_load(open(p)) if p.exists() else {}


def build(name):
    sc = load_scenarios()[name]
    fields = {k: v for k, v in sc.items() if k in ClaimMeta.__dataclass_fields__}
    meta = ClaimMeta(claim_id=f"DEMO-{name}", **fields)
    images = [config.SAMPLE_DIR / i for i in sc["images"]]
    return meta, images, config.SAMPLE_DIR / sc["report"], sc.get("title", name)
