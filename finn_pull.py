"""
FINN - 'Pull' mode research extractor (PRIMARY SOURCES ONLY).

Input : chunks of annual-report text and earnings-call (concall) transcripts.
Output: one structured JSON across 8 research categories. Every number carries
        an `evidence` block (doc_id, source_type, period, page/speaker, quote)
        so each claim can be traced back to a primary document.

    1. business_snapshot      what the company does (from the annual report)
    2. financial_snapshot     revenue / EBITDA / PAT, margins, growth
    3. debt_profile           borrowings, net debt, leverage, interest cover
    4. cash_flow_quality      CFO vs EBITDA/PAT, FCF, receivable days
    5. capital_allocation     capex, dividends, buybacks, acquisitions
    6. management_guidance    forward-looking statements (management speakers only)
    7. guidance_vs_delivery   last 4 quarters: guided vs actual + a worked example
    8. governance_red_flags   audit opinion, contingent liabilities, RPTs, pledges

Rules enforced in code:
  * Chunks whose source_type isn't annual_report / concall_transcript are rejected.
  * Analyst and moderator lines are never treated as guidance or actuals.
  * Anything not found is listed under `missing` instead of being guessed.

The extractors are rule-based (regex) so results are deterministic and auditable.
An LLM extraction pass can be swapped in per category later; keep the same
Fact/Evidence shape so the provenance guarantee stays.

Usage:
    python finn_pull.py                       # run on the built-in mock company
    python finn_pull.py --input chunks.jsonl  # your own chunks (one JSON object per line)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

ALLOWED_SOURCES = {"annual_report", "concall_transcript"}


# --------------------------------------------------------------------------- #
# 1. Data model
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    doc_id: str
    source_type: str            # annual_report | concall_transcript
    company: str
    period: str                 # "FY26" for an AR, "Q2FY27" for a concall
    text: str
    page: int | None = None
    speaker: str | None = None
    speaker_role: str | None = None   # management | analyst | moderator (transcripts)


@dataclass
class Evidence:
    doc_id: str
    source_type: str
    period: str
    quote: str
    page: int | None = None
    speaker: str | None = None


@dataclass
class Fact:
    value: float | str | None
    unit: str | None = None
    prior: float | None = None          # previous-year comparative where reported
    evidence: Evidence | None = None
    derived_from: list[str] | None = None   # for computed metrics


# --------------------------------------------------------------------------- #
# 2. Parsing raw documents into chunks
# --------------------------------------------------------------------------- #
def parse_annual_report(raw: str, doc_id: str, company: str, period: str) -> list[Chunk]:
    """Split on page markers ('--- page N ---' or form feeds), then on blank lines."""
    chunks, page = [], 1
    for block in re.split(r"(\f|^-{2,}\s*page\s*\d+\s*-{2,}$)", raw, flags=re.I | re.M):
        m = re.match(r"-{2,}\s*page\s*(\d+)", block or "", re.I)
        if m:
            page = int(m.group(1)); continue
        if block == "\f":
            page += 1; continue
        for para in re.split(r"\n\s*\n", block or ""):
            if para.strip():
                chunks.append(Chunk(doc_id, "annual_report", company, period, para.strip(), page=page))
    return chunks


def parse_transcript(raw: str, doc_id: str, company: str, period: str) -> list[Chunk]:
    """'Speaker: text' lines. Management names come from a 'Management:' header line,
    e.g.  Management: Ramesh Iyer (Managing Director), Anita Desai (CFO)"""
    mgmt: set[str] = set()
    header = re.search(r"^management\s*:\s*(.+)$", raw, re.I | re.M)
    if header:
        mgmt = {re.sub(r"\(.*?\)", "", n).strip() for n in header.group(1).split(",")}

    chunks, cur_spk, buf = [], None, []

    def flush():
        if cur_spk and buf:
            role = ("moderator" if cur_spk.lower().startswith("moderator")
                    else "management" if cur_spk in mgmt else "analyst")
            chunks.append(Chunk(doc_id, "concall_transcript", company, period,
                                " ".join(buf).strip(), speaker=cur_spk, speaker_role=role))

    for line in raw.splitlines():
        if re.match(r"^management\s*:", line, re.I):
            continue
        m = re.match(r"^([A-Z][A-Za-z.\s]{1,40}?)(?:\s*\(.*?\))?\s*:\s*(.*)$", line)
        if m:
            flush(); cur_spk, buf = m.group(1).strip(), [m.group(2)]
        elif line.strip():
            buf.append(line.strip())
    flush()
    return chunks


def enforce_primary_sources(chunks: list[Chunk]) -> tuple[list[Chunk], list[dict]]:
    ok, rejected = [], []
    for c in chunks:
        (ok if c.source_type in ALLOWED_SOURCES else rejected).append(c)
    return ok, [{"doc_id": c.doc_id, "source_type": c.source_type,
                 "reason": "not a primary source"} for c in rejected]


# --------------------------------------------------------------------------- #
# 3. Low-level helpers: sentences, numbers, periods
# --------------------------------------------------------------------------- #
def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z₹(])", text) if s.strip()]


_UNIT_TO_CR = {"crore": 1, "cr": 1, "lakh": 0.01, "bn": 100, "billion": 100, "million": 0.1}
MONEY_RE = re.compile(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(crore|cr|lakh|bn|billion|million)\b", re.I)
MONEY_RANGE_RE = re.compile(
    r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(?:-|–|to)\s*(?:₹\s*)?([\d,]+(?:\.\d+)?)\s*(crore|cr|lakh|bn|billion|million)\b", re.I)
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s?cent)", re.I)
PCT_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*(?:%|per\s?cent)", re.I)
TIMES_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:x|times)\b", re.I)
FLOOR_RE = re.compile(r"\b(at least|minimum of|upwards of|more than|over|north of)\s*$", re.I)


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def money_cr(m: re.Match, idx: int = 1) -> float:
    return round(_f(m.group(idx)) * _UNIT_TO_CR[m.group(m.lastindex).lower()], 2)


def parse_value(window: str, unit: str) -> tuple[float, float | None] | None:
    """Return (low, high) from the first number in `window`. high=None => floor ('at least')."""
    if unit == "%":
        rng, one = PCT_RANGE_RE.search(window), PCT_RE.search(window)
        if rng and (not one or rng.start() <= one.start()):
            return float(rng.group(1)), float(rng.group(2))
        if one:
            v = float(one.group(1))
            return (v, None) if FLOOR_RE.search(window[:one.start()]) else (v, v)
    else:  # ₹ crore
        rng, one = MONEY_RANGE_RE.search(window), MONEY_RE.search(window)
        if rng and (not one or rng.start() <= one.start()):
            k = _UNIT_TO_CR[rng.group(3).lower()]
            return round(_f(rng.group(1)) * k, 2), round(_f(rng.group(2)) * k, 2)
        if one:
            v = money_cr(one)
            return (v, None) if FLOOR_RE.search(window[:one.start()]) else (v, v)
    return None


# Indian fiscal periods: Q1 = Apr-Jun ... Q4 = Jan-Mar. Key = (fy, q); FY-only = (fy, 0)
Q_RE = re.compile(r"\bQ([1-4])\s*FY\s*'?(?:20)?(\d{2})\b", re.I)
FY_RE = re.compile(r"\bFY\s*'?(?:20)?(\d{2})\b", re.I)


def pkey(label: str) -> tuple[int, int]:
    m = Q_RE.search(label)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = FY_RE.search(label)
    return (int(m.group(1)), 0) if m else (0, 0)


def plabel(k: tuple[int, int]) -> str:
    return f"Q{k[1]}FY{k[0]:02d}" if k[1] else f"FY{k[0]:02d}"


def next_q(k: tuple[int, int]) -> tuple[int, int]:
    return (k[0], k[1] + 1) if k[1] < 4 else (k[0] + 1, 1)


def ordering(k: tuple[int, int]) -> tuple[int, int]:
    """Sort key where a full year sorts with its Q4 (when it is reported)."""
    return (k[0], 4 if k[1] == 0 else k[1])


def resolve_target(sent: str, call: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    """Which period does a forward-looking sentence refer to? Returns (key, inferred?)."""
    if (m := Q_RE.search(sent)):
        return (int(m.group(2)), int(m.group(1))), False
    if (m := FY_RE.search(sent)):
        return (int(m.group(1)), 0), False
    low = sent.lower()
    current_fy = call[0] + 1 if call[1] == 4 else call[0]   # Q4 call happens in the next FY
    if re.search(r"next year|next fiscal", low):
        return (current_fy + 1, 0), False
    if re.search(r"this year|full year|for the year|this fiscal|current (fiscal|year)", low):
        return (current_fy, 0), False
    if re.search(r"next quarter|coming quarter|upcoming quarter|current quarter|this quarter", low):
        return next_q(call), False
    return next_q(call), True       # default assumption, flagged


# --------------------------------------------------------------------------- #
# 4. Metric vocabulary for guidance / actuals in concalls
# --------------------------------------------------------------------------- #
METRICS = {
    "revenue_growth": (r"revenue(?:s)? (?:growth|grew|to grow|increased|growing)|top[- ]?line growth", "%", True),
    "ebitda_margin":  (r"ebitda margins?", "%", True),
    "volume_growth":  (r"volume(?:s)? (?:growth|grew|to grow)", "%", True),
    "pat_growth":     (r"(?:pat|net profit) (?:growth|grew)", "%", True),
    "capex":          (r"\bcapex\b|capital expenditure", "₹ cr", None),   # None: no 'better' direction
    "net_debt":       (r"net debt", "₹ cr", False),                        # lower is better
}
FORWARD_RE = re.compile(
    r"\b(we expect|we anticipate|expect(?:ing)?|guidance|guiding|we target|targeting|we aim|aiming|"
    r"should (?:be|come in)|will be in the range|outlook|confident of|we see|likely to)\b", re.I)


def metric_mentions(sent: str) -> list[tuple[str, float, float | None, str]]:
    found = []
    for name, (pat, unit, _) in METRICS.items():
        for m in re.finditer(pat, sent, re.I):
            v = parse_value(sent[m.end(): m.end() + 90], unit)
            if v:
                found.append((name, v[0], v[1], unit)); break
    return found


def _ev(c: Chunk, quote: str) -> Evidence:
    return Evidence(c.doc_id, c.source_type, c.period, quote[:240], c.page, c.speaker)


# --------------------------------------------------------------------------- #
# 5. Annual-report extractors (categories 1-5, 8)
# --------------------------------------------------------------------------- #
AR_LINES = {
    "revenue":       r"revenue from operations|total revenue",
    "ebitda":        r"\bebitda\b",
    "pat":           r"profit after tax|net profit|profit for the year",
    "cfo":           r"net cash (?:generated |flow )?from operating activities|cash from operations",
    "capex":         r"purchase of property,? plant and equipment|capital expenditure|\bcapex\b",
    "gross_debt":    r"total borrowings|gross debt",
    "cash":          r"cash and cash equivalents(?: and (?:other )?bank balances)?",
    "equity":        r"total equity|net worth",
    "finance_cost":  r"finance costs?|interest expense",
    "receivables":   r"trade receivables",
    "dividend_total": r"total dividend (?:payout|outgo)|dividend payout of",
    "buyback":       r"buy-?back",
    "contingent":    r"contingent liabilities",
    "rpt":           r"related party transactions",
}


def find_money_line(chunks: list[Chunk], label: str) -> Fact | None:
    for c in chunks:
        for s in sentences(c.text) if "\n" not in c.text else c.text.splitlines():
            m = re.search(label, s, re.I)
            if not m:
                continue
            vals = [money_cr(x) for x in MONEY_RE.finditer(s[m.end(): m.end() + 160])]
            if vals:
                return Fact(vals[0], "₹ cr", vals[1] if len(vals) > 1 else None, _ev(c, s.strip()))
    return None


def find_text(chunks: list[Chunk], pattern: str) -> tuple[str, Evidence] | None:
    for c in chunks:
        for s in sentences(c.text):
            if re.search(pattern, s, re.I):
                return s, _ev(c, s)
    return None


def ratio(a: Fact | None, b: Fact | None, name: str, scale=1.0, digits=2, use_prior=False) -> Fact | None:
    av = a and (a.prior if use_prior else a.value)
    bv = b and (b.prior if use_prior else b.value)
    if av is None or not bv:
        return None
    return Fact(round(av / bv * scale, digits), None, derived_from=[name])


def facts_to_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, Fact):
            out[k] = {kk: vv for kk, vv in asdict(v).items() if vv is not None}
        elif isinstance(v, dict):
            out[k] = facts_to_dict(v)
        else:
            out[k] = v
    return out


def ar_categories(ar: list[Chunk]) -> tuple[dict, list[str]]:
    missing: list[str] = []
    F = {}
    for key, pat in AR_LINES.items():
        F[key] = find_money_line(ar, pat)
        if F[key] is None:
            missing.append(key)

    def pct(fact_a, fact_b, prior=False):
        r = ratio(fact_a, fact_b, "", 100, 1, prior)
        return r.value if r else None

    def growth(f):
        return round((f.value / f.prior - 1) * 100, 1) if f and f.prior else None

    # 1. business snapshot
    desc = find_text(ar, r"\b(is engaged in|is a leading|is one of (?:the|India)|manufactures|operates)\b")
    business = {"description": Fact(desc[0], evidence=desc[1]) if desc else None}
    seg = find_text(ar, r"\bsegments?\b.*\b(comprise|consist|operates in)\b")
    if seg:
        business["segments"] = Fact(seg[0], evidence=seg[1])

    # 2. financial snapshot
    fin = {k: F[k] for k in ("revenue", "ebitda", "pat")}
    fin["ebitda_margin_pct"] = Fact(pct(F["ebitda"], F["revenue"]), "%", pct(F["ebitda"], F["revenue"], True),
                                    derived_from=["ebitda", "revenue"])
    fin["pat_margin_pct"] = Fact(pct(F["pat"], F["revenue"]), "%", pct(F["pat"], F["revenue"], True),
                                 derived_from=["pat", "revenue"])
    fin["revenue_growth_pct"] = Fact(growth(F["revenue"]), "%", derived_from=["revenue (current vs prior)"])
    fin["pat_growth_pct"] = Fact(growth(F["pat"]), "%", derived_from=["pat (current vs prior)"])

    # 3. debt profile
    debt = {k: F[k] for k in ("gross_debt", "cash", "equity", "finance_cost")}
    flags = []
    if F["gross_debt"] and F["cash"]:
        nd, ndp = F["gross_debt"].value - F["cash"].value, None
        if F["gross_debt"].prior is not None and F["cash"].prior is not None:
            ndp = round(F["gross_debt"].prior - F["cash"].prior, 2)
        debt["net_debt"] = Fact(round(nd, 2), "₹ cr", ndp, derived_from=["gross_debt", "cash"])
        if F["ebitda"]:
            nde = round(nd / F["ebitda"].value, 2)
            debt["net_debt_to_ebitda"] = Fact(nde, "x", derived_from=["net_debt", "ebitda"])
            if nde > 3:
                flags.append(f"Net debt/EBITDA {nde}x > 3x")
    if (de := ratio(F["gross_debt"], F["equity"], "gross_debt/equity")):
        debt["debt_to_equity"] = Fact(de.value, "x", derived_from=["gross_debt", "equity"])
    if (ic := ratio(F["ebitda"], F["finance_cost"], "ebitda/finance_cost", digits=1)):
        debt["interest_coverage"] = Fact(ic.value, "x", derived_from=["ebitda", "finance_cost"])
        if ic.value < 3:
            flags.append(f"Interest coverage {ic.value}x < 3x")
    debt["flags"] = flags

    # 4. cash-flow quality
    cash = {"cfo": F["cfo"], "capex": F["capex"]}
    cflags = []
    if (r := ratio(F["cfo"], F["ebitda"], "cfo/ebitda")):
        cash["cfo_to_ebitda"] = Fact(r.value, "x", ratio(F["cfo"], F["ebitda"], "", use_prior=True).value
                                     if F["cfo"].prior and F["ebitda"].prior else None, derived_from=["cfo", "ebitda"])
        if r.value < 0.7:
            cflags.append(f"CFO/EBITDA {r.value} < 0.7 — weak cash conversion")
    if (r := ratio(F["cfo"], F["pat"], "cfo/pat")):
        cash["cfo_to_pat"] = Fact(r.value, "x", derived_from=["cfo", "pat"])
        if r.value < 1:
            cflags.append(f"CFO/PAT {r.value} < 1 — profits not fully backed by cash")
    if F["cfo"] and F["capex"]:
        fcf = round(F["cfo"].value - F["capex"].value, 2)
        fcfp = (round(F["cfo"].prior - F["capex"].prior, 2)
                if F["cfo"].prior is not None and F["capex"].prior is not None else None)
        cash["free_cash_flow"] = Fact(fcf, "₹ cr", fcfp, derived_from=["cfo", "capex"])
        if fcf < 0:
            cflags.append("Negative free cash flow")
    if F["receivables"] and F["revenue"]:
        d = round(F["receivables"].value / F["revenue"].value * 365, 1)
        dp = (round(F["receivables"].prior / F["revenue"].prior * 365, 1)
              if F["receivables"].prior and F["revenue"].prior else None)
        cash["receivable_days"] = Fact(d, "days", dp, derived_from=["receivables", "revenue"])
        if dp is not None and d - dp > 10:
            cflags.append(f"Receivable days up {d - dp:.1f} YoY ({dp} → {d}) — check revenue quality")
    cash["flags"] = cflags

    # 5. capital allocation
    dps = find_text(ar, r"dividend of ₹\s*[\d.]+\s*per (equity )?share")
    alloc = {"capex": F["capex"], "dividend_total": F["dividend_total"], "buyback": F["buyback"]}
    if dps:
        v = re.search(r"₹\s*([\d.]+)\s*per", dps[0])
        alloc["dividend_per_share"] = Fact(float(v.group(1)), "₹/share", evidence=dps[1])
    if (p := ratio(F["dividend_total"], F["pat"], "", 100, 1)):
        alloc["dividend_payout_pct"] = Fact(p.value, "%", derived_from=["dividend_total", "pat"])
    if (acq := find_text(ar, r"\b(acquired|acquisition of)\b")):
        alloc["acquisitions"] = Fact(acq[0], evidence=acq[1])
    if F["capex"] and F["revenue"]:
        alloc["capex_to_revenue_pct"] = Fact(round(F["capex"].value / F["revenue"].value * 100, 1), "%",
                                             derived_from=["capex", "revenue"])

    # 8. governance / red flags
    gov, gflags = {}, []
    if (op := find_text(ar, r"\b(unmodified|unqualified|qualified|adverse|disclaimer of) opinion\b")):
        kind = re.search(r"(unmodified|unqualified|qualified|adverse|disclaimer of)", op[0], re.I).group(1).lower()
        gov["audit_opinion"] = Fact(kind, evidence=op[1])
        if kind in ("qualified", "adverse", "disclaimer of"):
            gflags.append(f"Auditor issued a {kind} opinion")
    if (em := find_text(ar, r"emphasis of matter")):
        gov["emphasis_of_matter"] = Fact(em[0], evidence=em[1])
        gflags.append("Emphasis of matter in audit report")
    if (ka := find_text(ar, r"(resignation|change) of (the )?statutory auditors?")):
        gov["auditor_change"] = Fact(ka[0], evidence=ka[1])
        gflags.append("Statutory auditor change")
    gov["contingent_liabilities"] = F["contingent"]
    if F["contingent"] and F["equity"]:
        cl = round(F["contingent"].value / F["equity"].value * 100, 1)
        gov["contingent_to_networth_pct"] = Fact(cl, "%", derived_from=["contingent", "equity"])
        if cl > 10:
            gflags.append(f"Contingent liabilities {cl}% of net worth")
    gov["related_party_transactions"] = F["rpt"]
    if F["rpt"] and F["revenue"] and F["rpt"].value / F["revenue"].value > 0.05:
        gflags.append("Related-party transactions > 5% of revenue")
    if (pl := find_text(ar, r"pledged")):
        v = PCT_RE.search(pl[0])
        gov["promoter_pledge_pct"] = Fact(float(v.group(1)) if v else pl[0], "%" if v else None, evidence=pl[1])
        if v and float(v.group(1)) > 0:
            gflags.append(f"Promoter pledge {v.group(1)}%")
    gov["flags"] = gflags

    cats = {"business_snapshot": business, "financial_snapshot": fin, "debt_profile": debt,
            "cash_flow_quality": cash, "capital_allocation": alloc, "governance_red_flags": gov}
    return cats, missing


# --------------------------------------------------------------------------- #
# 6. Concall extractors (categories 6-7)
# --------------------------------------------------------------------------- #
def extract_guidance_and_actuals(calls: list[Chunk]) -> tuple[list[dict], list[dict]]:
    guidance, actuals = [], []
    for c in calls:
        if c.speaker_role != "management":
            continue                                   # analysts' numbers are not guidance
        call = pkey(c.period)
        for s in sentences(c.text):
            mentions = metric_mentions(s)
            if not mentions:
                continue
            if FORWARD_RE.search(s):
                target, inferred = resolve_target(s, call)
                for name, lo, hi, unit in mentions:
                    guidance.append({"metric": name, "unit": unit, "low": lo, "high": hi,
                                     "target_period": plabel(target), "target_inferred": inferred,
                                     "given_in": c.period, "evidence": asdict(_ev(c, s))})
            else:
                q, fy = Q_RE.search(s), FY_RE.search(s)
                per = (int(q.group(2)), int(q.group(1))) if q else (int(fy.group(1)), 0) if fy else call
                for name, lo, _, unit in mentions:
                    actuals.append({"metric": name, "unit": unit, "value": lo,
                                    "period": plabel(per), "reported_in": c.period,
                                    "evidence": asdict(_ev(c, s))})
    return guidance, actuals


def detect_revisions(guidance: list[dict]) -> list[dict]:
    by: dict[tuple, list[dict]] = {}
    for g in guidance:
        by.setdefault((g["metric"], g["target_period"]), []).append(g)
    out = []
    for (metric, tgt), gs in by.items():
        gs.sort(key=lambda g: ordering(pkey(g["given_in"])))
        for a, b in zip(gs, gs[1:]):
            if (a["low"], a["high"]) != (b["low"], b["high"]):
                mid = lambda g: (g["low"] + (g["high"] if g["high"] is not None else g["low"])) / 2
                out.append({"metric": metric, "target_period": tgt,
                            "from": [a["low"], a["high"]], "from_call": a["given_in"],
                            "to": [b["low"], b["high"]], "to_call": b["given_in"],
                            "direction": "raised" if mid(b) > mid(a) else "cut"})
    return out


def verdict(g: dict, actual: float) -> tuple[str, float | None]:
    _, unit, higher_better = METRICS[g["metric"]]
    lo, hi = g["low"], g["high"]
    if lo == hi:                                      # point guidance -> tolerance band
        tol = 0.5 if unit == "%" else abs(lo) * 0.05
        lo, hi = lo - tol, hi + tol
    if hi is None:                                    # floor ("at least")
        return ("met" if actual >= lo else "missed"), round(actual - lo, 2)
    if lo <= actual <= hi:
        return "met", 0.0
    dev = round(actual - hi, 2) if actual > hi else round(actual - lo, 2)
    if higher_better is None:
        return ("above range" if dev > 0 else "below range"), dev
    good = (dev > 0) == higher_better
    return ("beat" if good else "missed"), dev


def guidance_vs_delivery(guidance: list[dict], actuals: list[dict], fin: dict,
                         ar_period: str | None, n_quarters: int = 4) -> dict:
    act_quarters = sorted({pkey(a["period"]) for a in actuals if pkey(a["period"])[1]}, key=ordering)
    window = act_quarters[-n_quarters:]
    in_window = lambda k: k in window or (k[1] == 0 and (k[0], 4) in window)

    rows = []
    for a in actuals:
        tk = pkey(a["period"])
        if not in_window(tk):
            continue
        prior = [g for g in guidance if g["metric"] == a["metric"] and g["target_period"] == a["period"]
                 and ordering(pkey(g["given_in"])) < ordering(pkey(a["reported_in"]))]
        if not prior:
            continue
        prior.sort(key=lambda g: ordering(pkey(g["given_in"])))
        g, first = prior[-1], prior[0]
        v, dev = verdict(g, a["value"])
        row = {"metric": a["metric"], "period": a["period"], "unit": a["unit"],
               "guided": [g["low"], g["high"]], "guided_in": g["given_in"],
               "actual": a["value"], "reported_in": a["reported_in"],
               "verdict": v, "deviation": dev,
               "guidance_evidence": g["evidence"], "actual_evidence": a["evidence"]}
        if first is not g:
            row["original_guidance"] = {"range": [first["low"], first["high"]], "given_in": first["given_in"]}
        # cross-check full-year revenue growth against the audited annual report
        if a["metric"] == "revenue_growth" and tk[1] == 0 and ar_period == a["period"]:
            ar_g = fin.get("revenue_growth_pct")
            if ar_g and ar_g.value is not None:
                row["annual_report_cross_check"] = {"value": ar_g.value,
                                                    "matches": abs(ar_g.value - a["value"]) <= 0.3}
        rows.append(row)

    rows.sort(key=lambda r: (ordering(pkey(r["period"])), r["metric"]))
    scored = [r for r in rows if r["verdict"] in ("met", "beat", "missed")]
    hits = sum(r["verdict"] in ("met", "beat") for r in scored)
    return {
        "window_quarters": [plabel(k) for k in window],
        "records": rows,
        "scorecard": {"scored": len(scored), "met_or_beat": hits,
                      "missed": len(scored) - hits,
                      "hit_rate_pct": round(hits / len(scored) * 100, 1) if scored else None},
        "worked_example": worked_example(rows),
    }


def worked_example(rows: list[dict]) -> dict | None:
    """Step-by-step walk-through of one guided-vs-actual comparison (a miss if there is one)."""
    if not rows:
        return None
    r = next((x for x in rows if x["verdict"] == "missed"), rows[-1])
    u = "%" if r["unit"] == "%" else " ₹ cr"
    lo, hi = r["guided"]
    rng = f"{lo}{u} or higher" if hi is None else (f"~{lo}{u}" if lo == hi else f"{lo}–{hi}{u}")
    band = ("point guidance, so a ±0.5pp tolerance band is applied" if lo == hi and r["unit"] == "%"
            else "point guidance, so a ±5% tolerance band is applied" if lo == hi
            else "a floor: anything at or above it counts as met" if hi is None
            else "an explicit range")
    mid = lo if hi is None else (lo + hi) / 2
    pp = "pp" if r["unit"] == "%" else " ₹ cr"
    steps = [
        f"1. What management said ({r['guided_in']} call, {r['guidance_evidence'].get('speaker')}): "
        f"\"{r['guidance_evidence']['quote']}\"",
        f"2. Parsed target: {r['metric']} for {r['period']} = {rng} ({band}).",
        f"3. What was delivered ({r['reported_in']} call): \"{r['actual_evidence']['quote']}\" → {r['actual']}{u}.",
        f"4. Comparison: actual {r['actual']}{u} vs guided {rng}. "
        f"Gap to guided midpoint: {r['actual'] - mid:+.2f}{pp}; "
        f"gap to the nearest edge of the scoring band: {r['deviation']:+}{pp}.",
        f"5. Verdict: {r['verdict'].upper()}.",
    ]
    if "original_guidance" in r:
        o = r["original_guidance"]
        steps.append(f"6. Note: guidance was first given as {o['range']} in {o['given_in']} and revised "
                     f"before delivery; the latest pre-delivery guidance is the one scored.")
    return {"metric": r["metric"], "period": r["period"], "verdict": r["verdict"], "steps": steps}


# --------------------------------------------------------------------------- #
# 7. Orchestrator
# --------------------------------------------------------------------------- #
def build_research(chunks: list[Chunk]) -> dict:
    chunks, rejected = enforce_primary_sources(chunks)
    ar = [c for c in chunks if c.source_type == "annual_report"]
    calls = [c for c in chunks if c.source_type == "concall_transcript"]
    ar_period = max((c.period for c in ar), key=pkey, default=None)
    ar_latest = [c for c in ar if c.period == ar_period]

    cats, missing = ar_categories(ar_latest) if ar_latest else ({}, ["annual_report"])
    guidance, actuals = extract_guidance_and_actuals(calls)

    latest_call = max((pkey(c.period) for c in calls), key=ordering, default=None)
    latest_by_target: dict[tuple, dict] = {}
    for g in sorted(guidance, key=lambda g: ordering(pkey(g["given_in"]))):
        latest_by_target[(g["metric"], g["target_period"])] = g     # later calls supersede
    outstanding = [g for g in latest_by_target.values()
                   if latest_call and ordering(pkey(g["target_period"])) > ordering(latest_call)]
    cats["management_guidance"] = {
        "outstanding": outstanding,
        "revisions": detect_revisions(guidance),
        "all_guidance_count": len(guidance),
    }
    cats["guidance_vs_delivery"] = guidance_vs_delivery(
        guidance, actuals, cats.get("financial_snapshot", {}), ar_period)

    order = ["business_snapshot", "financial_snapshot", "debt_profile", "cash_flow_quality",
             "capital_allocation", "management_guidance", "guidance_vs_delivery", "governance_red_flags"]
    return {
        "company": (chunks[0].company if chunks else None),
        "sources_used": sorted({(c.doc_id, c.source_type, c.period) for c in chunks}),
        "rejected_sources": rejected,
        "annual_report_period": ar_period,
        "missing_fields": missing,
        "categories": {k: facts_to_dict(cats[k]) if k in cats else None for k in order},
    }


def load_chunks_jsonl(path: str) -> list[Chunk]:
    return [Chunk(**json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# 8. Mock primary documents (FICTIONAL company, for testing only)
# --------------------------------------------------------------------------- #
MOCK_AR = """--- page 4 ---
Kaveri Fibres Limited is engaged in the manufacture of viscose and specialty cellulosic fibres, with three plants in Karnataka and Gujarat. Its business segments comprise Fibre, Yarn and Specialty Chemicals.

--- page 38 ---
Revenue from operations: ₹4,820 crore (FY25: ₹4,230 crore)
EBITDA: ₹868 crore (FY25: ₹740 crore)
Profit after tax: ₹452 crore (FY25: ₹398 crore)
Finance costs: ₹96 crore (FY25: ₹104 crore)

--- page 112 ---
Total borrowings: ₹1,150 crore (FY25: ₹1,240 crore)
Cash and cash equivalents and other bank balances: ₹310 crore (FY25: ₹260 crore)
Total equity: ₹3,050 crore (FY25: ₹2,700 crore)
Trade receivables: ₹1,020 crore (FY25: ₹760 crore)

--- page 118 ---
Net cash generated from operating activities: ₹610 crore (FY25: ₹520 crore)
Purchase of property, plant and equipment: ₹380 crore (FY25: ₹290 crore)

--- page 9 ---
The Board recommended a final dividend of ₹4.00 per equity share. Total dividend payout of ₹118 crore.
During the year the Company completed the acquisition of a 60% stake in a lyocell start-up.

--- page 96 ---
In our opinion, the standalone financial statements give a true and fair view; we have issued an unmodified opinion. We draw attention to the emphasis of matter relating to a pending excise duty dispute.

--- page 160 ---
Contingent liabilities not acknowledged as debts: ₹210 crore (FY25: ₹180 crore)
Related party transactions during the year aggregated ₹145 crore (FY25: ₹120 crore)
None of the promoter shareholding is pledged; pledged shares 0% of promoter holding.
"""

MOCK_CALLS = {
    "Q2FY26": """Management: Ramesh Iyer (Managing Director), Anita Desai (Chief Financial Officer)
Moderator: Welcome to the Q2 FY26 earnings call of Kaveri Fibres.
Ramesh Iyer: For Q3 FY26, we expect revenue growth of 12-14% YoY. We expect EBITDA margin of around 18% in the coming quarter. For FY26 we are guiding revenue growth of 13-15%.
Anita Desai: Capex for FY26 should be in the range of ₹350-400 crore.
Vikram Shah: Do you expect EBITDA margin of 25% next year?
Ramesh Iyer: We would not like to comment that far out.""",
    "Q3FY26": """Management: Ramesh Iyer (Managing Director), Anita Desai (Chief Financial Officer)
Ramesh Iyer: Revenue grew 12.8% YoY this quarter. EBITDA margin came in at 17.2% due to higher pulp costs.
Anita Desai: For Q4 FY26 we expect revenue growth of 14-16% and EBITDA margin of 18-19%.""",
    "Q4FY26": """Management: Ramesh Iyer (Managing Director), Anita Desai (Chief Financial Officer)
Ramesh Iyer: Revenue grew 16.4% YoY in the quarter. EBITDA margin stood at 18.6%.
Anita Desai: For FY26, revenue grew 14.0% and capex was ₹380 crore.
Ramesh Iyer: For Q1 FY27, we expect revenue growth of 10-12% and EBITDA margin of at least 18%. For this year we are guiding revenue growth of 14-16%.""",
    "Q1FY27": """Management: Ramesh Iyer (Managing Director), Anita Desai (Chief Financial Officer)
Ramesh Iyer: Revenue grew 9.1% YoY, impacted by monsoon disruption at the Gujarat plant. EBITDA margin was at 17.9%.
Anita Desai: Next quarter we expect revenue growth of 12-13% and EBITDA margin of about 18.5%.""",
    "Q2FY27": """Management: Ramesh Iyer (Managing Director), Anita Desai (Chief Financial Officer)
Ramesh Iyer: Revenue grew 12.6% YoY. EBITDA margin came in at 18.9%.
Anita Desai: For Q3 FY27, we expect revenue growth of 13-15%. We remain confident of FY27 revenue growth of 13-15%.""",
}


def mock_chunks() -> list[Chunk]:
    co = "Kaveri Fibres Ltd (fictional)"
    chunks = parse_annual_report(MOCK_AR, "KAVERI-AR-FY26", co, "FY26")
    for per, raw in MOCK_CALLS.items():
        chunks += parse_transcript(raw, f"KAVERI-CC-{per}", co, per)
    # a non-primary source that must be rejected
    chunks.append(Chunk("BROKER-NOTE-01", "broker_report", co, "Q2FY27",
                        "We expect revenue growth of 20% in FY27."))
    return chunks


# --------------------------------------------------------------------------- #
# 9. CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="FINN Pull-mode research extractor")
    ap.add_argument("--input", help="JSONL of Chunk objects; omit to use the mock company")
    ap.add_argument("--out", default="data/research.json")
    args = ap.parse_args()

    res = build_research(load_chunks_jsonl(args.input) if args.input else mock_chunks())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))

    c = res["categories"]
    print(f"[FINN Pull] {res['company']} | AR {res['annual_report_period']} | "
          f"{len(res['sources_used'])} primary docs | rejected {len(res['rejected_sources'])} | "
          f"missing {res['missing_fields']}")
    for k in ("debt_profile", "cash_flow_quality", "governance_red_flags"):
        print(f"  {k} flags: {c[k]['flags']}")
    gvd = c["guidance_vs_delivery"]
    print(f"\nGuidance vs delivery ({', '.join(gvd['window_quarters'])}) — hit rate "
          f"{gvd['scorecard']['hit_rate_pct']}% ({gvd['scorecard']['met_or_beat']}/{gvd['scorecard']['scored']})")
    for r in gvd["records"]:
        print(f"  {r['period']:<7} {r['metric']:<15} guided {r['guided']} ({r['guided_in']}) "
              f"actual {r['actual']:<6} → {r['verdict']}")
    print("\nRevisions:", c["management_guidance"]["revisions"])
    print("\nWorked example:")
    for s in gvd["worked_example"]["steps"]:
        print("  " + s)
    print(f"\n[FINN Pull] saved {args.out}")


if __name__ == "__main__":
    main()
