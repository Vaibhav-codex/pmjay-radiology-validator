"""Canonical finding vocabulary + indicative terminology mapping (SNOMED-CT / ICD-10).
Mappings are indicative only - validate against the licensed terminology service in production."""

CANONICAL = {
    "fracture":         dict(display="Fracture",            snomed="125605004", icd10="S00-T14 (site-specific)", critical=True),
    "infiltrate":       dict(display="Infiltrate / consolidation", snomed="233604007", icd10="J18.9", critical=False),
    "effusion":         dict(display="Pleural effusion",    snomed="60046008",  icd10="J90",   critical=True),
    "mass":             dict(display="Mass / tumour",       snomed="108369006", icd10="D49.9", critical=True),
    "fluid_collection": dict(display="Fluid collection",    snomed="389026000", icd10="R18.8", critical=False),
    "osteoarthritis":   dict(display="Osteoarthritis",      snomed="396275006", icd10="M17.9", critical=False),
    "calculus":         dict(display="Calculus / stones",   snomed="235919008", icd10="K80.20", critical=False),
}

# what each canonical label would normally be visible on (used to avoid false 'unsupported' flags)
def display(label: str) -> str:
    return CANONICAL.get(label, {}).get("display", label.replace("_", " ").title())
