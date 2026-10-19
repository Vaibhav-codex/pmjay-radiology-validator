"""OPTIONAL: transformer-based clinical NER (ClinicalBERT / BioLinkBERT / BioMistral).
Not enabled by default. Fine-tune a token-classification model on your labels
(FINDING, SEVERITY, SITE, LATERALITY, PROCEDURE) and map its output into ReportEntities.
The rule-based extractor remains the fallback and the source of negation handling."""
from .entities import RuleBasedExtractor


class TransformerExtractor(RuleBasedExtractor):
    name = "transformer+rules"

    def __init__(self, model_name_or_path: str, **kw):
        super().__init__(**kw)
        from transformers import pipeline          # raises ImportError if not installed
        self.ner = pipeline("token-classification", model=model_name_or_path, aggregation_strategy="simple")

    def extract(self, text, ocr_confidence=None):
        ent = super().extract(text, ocr_confidence)
        # TODO: run self.ner(text), merge entities not already found by the lexicon
        ent.extractor = self.name
        return ent
