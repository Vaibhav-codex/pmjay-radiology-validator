"""Rule-based clinical entity extraction with negation, severity, laterality and site attachment.
Drop-in replaceable by a transformer NER (see transformer_ner.py) that returns the same ReportEntities."""
import re
import yaml
from .. import config
from ..schemas import ReportEntities, ReportFinding, ReportProcedure

POST_NEG = {"absent", "unremarkable", "not seen", "not visualized", "not identified", "negative", "nil"}
CLAUSE_BREAK = re.compile(r"\b(but|however|except|although|though|apart from)\b|[;:]")
DUR = re.compile(r"(\d+)\s*(day|days|week|weeks|month|months)\b")


def _compile(terms):
    terms = sorted({t.lower() for t in terms}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in terms) + r")(?![a-z0-9])")


class RuleBasedExtractor:
    name = "rule-based"

    def __init__(self, lexicon_path=None):
        lex = yaml.safe_load(open(lexicon_path or config.LEXICON_PATH))
        self.lex = lex
        self.finding_map = {t.lower(): lab for lab, ts in lex["findings"].items() for t in ts}
        self.finding_re = _compile(self.finding_map)
        self.proc_map = {t.lower(): p for p, ts in lex["procedures"].items() for t in ts}
        self.proc_re = _compile(self.proc_map)
        self.sev_map = {t.lower(): s for s, ts in lex["severity"].items() for t in ts}
        self.sev_re = _compile(self.sev_map)
        self.neg_re = _compile(lex["negation"])
        self.lat_map = {t.lower(): s for s, ts in lex["laterality"].items() for t in ts}
        self.lat_re = _compile(self.lat_map)
        self.site_map = {t.lower(): s for s, ts in lex["sites"].items() for t in ts}
        self.site_re = _compile(self.site_map)
        self.cons_re = _compile(lex["conservative_cues"])

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _sentences(text):
        out, pos = [], 0
        for m in re.finditer(r"[^.\n;]+[.\n;]?", text):
            if m.group().strip():
                out.append((m.start(), m.group()))
        return out

    @staticmethod
    def _nearest(regex, mapping, sent_l, start, end, before=60, after=40):
        best, bd = None, 10 ** 9
        for m in regex.finditer(sent_l):
            if m.start() >= start and m.end() <= end:
                continue
            d = start - m.end() if m.end() <= start else m.start() - end
            lim = before if m.end() <= start else after
            if 0 <= d <= lim and d < bd:
                best, bd = mapping[m.group(1)], d
        return best

    def _negated(self, sent_l, start, end):
        pre = sent_l[:start]
        cuts = list(CLAUSE_BREAK.finditer(pre))
        if cuts:
            pre = pre[cuts[-1].end():]
        pre_tail = " ".join(pre.split()[-6:])
        if self.neg_re.search(pre_tail):
            return True
        post_tail = " ".join(sent_l[end:].split()[:4])
        m = self.neg_re.search(post_tail)
        return bool(m and m.group(1) in POST_NEG)

    # -- main ---------------------------------------------------------------
    def extract(self, text: str, ocr_confidence=None) -> ReportEntities:
        findings, procs, sites = [], [], set()
        conservative_days = None
        for off, sent in self._sentences(text):
            sl = sent.lower()
            for m in self.site_re.finditer(sl):
                sites.add(self.site_map[m.group(1)])
            for m in self.finding_re.finditer(sl):
                label = self.finding_map[m.group(1)]
                s, e = m.start(), m.end()
                findings.append(ReportFinding(
                    label=label, text=sent[s:e], negated=self._negated(sl, s, e),
                    severity=self._nearest(self.sev_re, self.sev_map, sl, s, e, 50, 30) or "unknown",
                    laterality=self._nearest(self.lat_re, self.lat_map, sl, s, e, 60, 40) or "unknown",
                    site=self._nearest(self.site_re, self.site_map, sl, s, e, 80, 60) or "unknown",
                    sentence=sent.strip(), start=off + s, end=off + e))
            for m in self.proc_re.finditer(sl):
                name = self.proc_map[m.group(1)]
                s, e = m.start(), m.end()
                if self._negated(sl, s, e):
                    continue
                if any(q.name == name and abs(q.start - (off + s)) < 60 for q in procs):
                    continue
                procs.append(ReportProcedure(
                    name=name, text=sent[s:e],
                    laterality=self._nearest(self.lat_re, self.lat_map, sl, s, e, 60, 40) or "unknown",
                    start=off + s, end=off + e))
            if self.cons_re.search(sl):
                for d in DUR.finditer(sl):
                    n, unit = int(d.group(1)), d.group(2)
                    days = n * (30 if unit.startswith("month") else 7 if unit.startswith("week") else 1)
                    conservative_days = max(conservative_days or 0, days)
        return ReportEntities(text=text, findings=findings, procedures=procs, sites=sorted(sites),
                              conservative_days=conservative_days, ocr_confidence=ocr_confidence)


def highlight_html(ent: ReportEntities) -> str:
    """Escape the report and wrap entity spans for the dashboard."""
    from html import escape
    spans = [(f.start, f.end, "neg" if f.negated else "find", f.label) for f in ent.findings]
    spans += [(p.start, p.end, "proc", p.name) for p in ent.procedures]
    spans.sort()
    out, pos = [], 0
    for s, e, kind, lab in spans:
        if s < pos:
            continue
        out.append(escape(ent.text[pos:s]))
        out.append(f'<mark class="ent {kind}" title="{escape(lab)}{" (negated)" if kind == "neg" else ""}">{escape(ent.text[s:e])}</mark>')
        pos = e
    out.append(escape(ent.text[pos:]))
    return "".join(out).replace("\n", "<br>")
