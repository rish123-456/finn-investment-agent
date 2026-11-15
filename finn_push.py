"""
FINN - 'Push' mode backend: 24-hour digest of corporate filings.

Pipeline:
    raw filings  ->  window filter (last N hours, IST)
                 ->  classify (rule-based)        -> drop ROUTINE compliance
                 ->  enrich (results QoQ/YoY, management-change details)
                 ->  score + dedupe per (ticker, category)
                 ->  digest JSON + short push-notification headlines

All filings here are SIMULATED (source="MOCK"). The people named in them are
made up, and the "LinkedIn bio" snippets are synthetic templates flagged
mock=True. Replace them with a real profile source before any user sees them.

Usage:
    python finn_push.py                  # build digest for "now", print + save JSON
    python finn_push.py --hours 48       # wider window
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# Re-use the universe/portfolio from the data layer if it's alongside us.
try:
    from finn_data import UNIVERSE, MOCK_PORTFOLIO
    PORTFOLIO_TICKERS = {h.ticker for h in MOCK_PORTFOLIO}
except ImportError:  # standalone fallback
    UNIVERSE = {}
    PORTFOLIO_TICKERS = set()


# --------------------------------------------------------------------------- #
# 1. Data model
# --------------------------------------------------------------------------- #
@dataclass
class Filing:
    id: str
    ticker: str
    company: str
    filed_at: datetime          # tz-aware, IST
    subject: str
    body: str
    source: str = "MOCK"        # later: "NSE" / "BSE"


@dataclass
class DigestItem:
    filing_id: str
    ticker: str
    company: str
    filed_at: str
    category: str
    priority: int
    headline: str               # <=120 chars, for the push notification
    subject: str
    in_portfolio: bool
    details: dict = field(default_factory=dict)
    merged_filing_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 2. Classification rules (order matters: first match wins)
# --------------------------------------------------------------------------- #
# Routine compliance paperwork - dropped from the digest.
ROUTINE_PATTERNS = [
    r"regulation\s*74\s*\(5\)",                 # DP regs compliance certificate
    r"regulation\s*7\s*\(3\)",                  # share transfer agent certificate
    r"regulation\s*40\s*\(9\)",
    r"trading window",                          # closure/opening of trading window
    r"loss of share certificate|duplicate share certificate",
    r"newspaper (publication|advertisement)",
    r"regulation\s*31\s*\(4\)",                 # SAST annual pledge disclosure
    r"investor complaints",
    r"change in (registered office )?address of (rta|registrar)",
    r"intimation of (analyst|investor) (meet|call)",
    r"certificate .*company secretary in practice",
]

CATEGORY_RULES: list[tuple[str, str]] = [
    ("RESULTS",     r"financial results|\bresults\b|\bq[1-4]\s*fy\d{2}"),
    ("M&A",         r"acquisition|amalgamation|merger|demerger|scheme of arrangement|stake (sale|purchase)|divest"),
    ("MGMT_CHANGE", r"resignation|appointment|cessation|re-?appoint|steps? down|change in (key managerial|directorate|management)"),
    ("LITIGATION",  r"litigation|show cause|penalty|order (from|of) (sebi|cci|tribunal|court)|raid|search operation"),
    ("ORDER_WIN",   r"order (win|received|bagged)|letter of award|contract (worth|valued)"),
    ("RATING",      r"credit rating|rating (upgrade|downgrade|reaffirm)"),
    ("CORP_ACTION", r"dividend|buy-?back|bonus|stock split|rights issue|record date"),
]

BASE_PRIORITY = {
    "RESULTS": 90, "M&A": 85, "MGMT_CHANGE": 70, "LITIGATION": 65,
    "ORDER_WIN": 60, "RATING": 55, "CORP_ACTION": 50, "OTHER": 30,
}

KMP_ROLES = ["Managing Director", "Chief Executive Officer", "Chief Financial Officer",
             "Chief Operating Officer", "Chairman", "Whole-time Director"]
OTHER_ROLES = ["Independent Director", "Non-Executive Director", "Company Secretary",
               "Compliance Officer", "Chief Technology Officer", "Chief Risk Officer"]
ROLE_ABBR = {"Managing Director": "MD", "Chief Executive Officer": "CEO",
             "Chief Financial Officer": "CFO", "Chief Operating Officer": "COO",
             "Whole-time Director": "WTD", "Chief Technology Officer": "CTO",
             "Chief Risk Officer": "CRO", "Company Secretary": "CS"}


def is_routine(f: Filing) -> bool:
    text = f"{f.subject} {f.body}".lower()
    return any(re.search(p, text) for p in ROUTINE_PATTERNS)


def classify(f: Filing) -> str:
    if is_routine(f):
        return "ROUTINE"
    text = f"{f.subject} {f.body[:400]}".lower()   # subject + lead is enough
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, text):
            return cat
    return "OTHER"


# --------------------------------------------------------------------------- #
# 3. Results extraction: QoQ / YoY from the standard SEBI results table
#    (columns: current quarter | preceding quarter | same quarter last year)
# --------------------------------------------------------------------------- #
METRIC_PATTERNS = {
    "revenue": r"revenue from operations|total income from operations|net interest income",
    "ebitda":  r"\bebitda\b|operating profit",
    "pat":     r"net profit|profit after tax|profit for the (period|quarter)",
    "eps":     r"\beps\b|earnings per share",
}


def _num(tok: str) -> float | None:
    tok = tok.strip().replace(",", "").replace("₹", "")
    neg = tok.startswith("(") and tok.endswith(")")
    tok = tok.strip("()")
    try:
        v = float(tok)
    except ValueError:
        return None
    return -v if neg else v


def _pct(cur: float, base: float) -> float | None:
    if base is None or cur is None or base == 0:
        return None
    if base < 0:            # loss in base period - % change is misleading
        return None
    return round((cur / base - 1) * 100, 1)


def extract_results(body: str) -> dict:
    """Parse pipe-delimited results rows -> {metric: {cur, prev_q, prev_y, qoq_pct, yoy_pct}}."""
    out: dict = {}
    period = re.search(r"quarter ended\s+([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{4})", body, re.I)
    unit = re.search(r"₹\s*in\s*(crore|lakh|million)", body, re.I)
    out["period_end"] = period.group(1) if period else None
    out["unit"] = f"₹ {unit.group(1).lower()}" if unit else None
    out["metrics"] = {}

    for line in body.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) < 4:
            continue
        label = cells[0].lower()
        for key, pat in METRIC_PATTERNS.items():
            if key in out["metrics"] or not re.search(pat, label):
                continue
            cur, prev_q, prev_y = (_num(c) for c in cells[1:4])
            if cur is None:
                continue
            m = {"current": cur, "prev_quarter": prev_q, "prev_year": prev_y,
                 "qoq_pct": _pct(cur, prev_q), "yoy_pct": _pct(cur, prev_y)}
            # flag sign flips explicitly instead of a nonsense %
            if prev_y is not None and prev_y < 0 <= cur:
                m["note"] = "turned profitable YoY"
            elif prev_y is not None and prev_y >= 0 > cur:
                m["note"] = "swung to loss YoY"
            out["metrics"][key] = m
            break
    return out


def _fmt_delta(p):
    return "n/a" if p is None else f"{p:+.1f}%"


def results_headline(ticker: str, r: dict) -> str:
    sym = ticker.split(".")[0]
    parts = []
    for key, label in (("revenue", "Rev"), ("pat", "PAT")):
        m = r["metrics"].get(key)
        if m:
            yoy = m.get("note") or f"{_fmt_delta(m['yoy_pct'])} YoY"
            qoq = f"{_fmt_delta(m['qoq_pct'])} QoQ" if m["qoq_pct"] is not None else None
            if qoq is None and m["prev_quarter"] is not None and m["prev_quarter"] < 0:
                qoq = "vs loss QoQ"
            parts.append(f"{label} ₹{m['current']:,.0f}cr ({', '.join(x for x in (qoq, yoy) if x)})")
    return (f"{sym} results: " + "; ".join(parts))[:120] if parts else f"{sym} announced results"


# --------------------------------------------------------------------------- #
# 4. Management-change extraction + mock bio
# --------------------------------------------------------------------------- #
NAME_RE = re.compile(r"\b(Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)+)")


def extract_mgmt_change(body: str) -> dict:
    low = body.lower()
    if re.search(r"resign|cessation|step(ped|s)? down|superannuat", low):
        kind = "resignation"
    elif re.search(r"re-?appoint", low):
        kind = "reappointment"
    elif "appoint" in low:
        kind = "appointment"
    else:
        kind = "change"

    name_m = NAME_RE.search(body)
    roles = sorted((r for r in KMP_ROLES + OTHER_ROLES if r.lower() in low), key=low.find)
    role = roles[0] if roles else None
    eff = re.search(r"(?:with effect from|effective(?: from)?)\s+("
                    r"[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]+,?\s+\d{4}|\d{1,2}[./-]\d{1,2}[./-]\d{4})",
                    body, re.I)
    reason = re.search(r"(?:due to|citing|on account of)\s+(.+?)(?=,?\s*(?:with effect|effective)|\.|$)", body, re.I)
    succ = re.search(r"successor|interim|additional charge|in (his|her) place", low)

    return {
        "type": kind,
        "person": f"{name_m.group(1)} {name_m.group(2)}" if name_m else None,
        "role": role,
        "all_roles": roles,
        "is_kmp": any(r in KMP_ROLES for r in roles),
        "effective": eff.group(1).strip() if eff else None,
        "reason": reason.group(1).strip() if reason else None,
        "succession_mentioned": bool(succ),
    }


def mock_linkedin_bio(person: str, role: str | None, company: str) -> dict:
    """Deterministic SYNTHETIC profile snippet - placeholder for a real lookup."""
    rng = random.Random(int(hashlib.md5(person.encode()).hexdigest(), 16))
    finance = role and ("Financial" in role or "Risk" in role)
    tech = role and "Technology" in role
    edu = rng.choice(["MBA (Finance), CA" if finance else "MBA, B.Tech",
                      "Chartered Accountant" if finance else "M.S., B.E.",
                      "PGDM, B.Com" if finance else "PGDM, B.Tech"])
    prev = rng.choice([
        "a large private-sector bank" if finance else "a global IT services firm",
        "a Big Four advisory practice" if finance else "a consumer goods MNC",
        "a listed NBFC" if finance else ("a SaaS unicorn" if tech else "an infrastructure conglomerate"),
    ])
    yrs = rng.randint(18, 32)
    title = role or "Senior leader"
    edu = edu.rstrip(".")
    return {
        "mock": True,
        "source": "synthetic template - replace with real profile lookup",
        "headline": f"{title} at {company}",
        "snippet": (f"{yrs}+ years across {('finance, treasury and investor relations' if finance else 'operations, strategy and P&L leadership')}. "
                    f"Previously held senior roles at {prev}. {edu}."),
        "experience_years": yrs,
    }


def mgmt_headline(ticker: str, m: dict) -> str:
    sym = ticker.split(".")[0]
    role = " & ".join(ROLE_ABBR.get(r, r) for r in m.get("all_roles") or []) or "official"
    who = m["person"] or "An official"
    verb = {"resignation": "resigns as", "appointment": "appointed", "reappointment": "re-appointed"}.get(m["type"], "change:")
    tail = f" ({m['reason']})" if m["type"] == "resignation" and m["reason"] else ""
    return f"{sym}: {who} {verb} {role}{tail}"[:120]


# --------------------------------------------------------------------------- #
# 5. Scoring
# --------------------------------------------------------------------------- #
def score(cat: str, details: dict, in_portfolio: bool) -> int:
    p = BASE_PRIORITY.get(cat, 30)
    if cat == "RESULTS":
        pat = details.get("metrics", {}).get("pat", {})
        if pat.get("note") or abs(pat.get("yoy_pct") or 0) >= 15:
            p += 5                                   # big surprise either way
    if cat == "MGMT_CHANGE":
        if details.get("is_kmp"):
            p += 12
            if details.get("type") == "resignation" and not details.get("succession_mentioned"):
                p += 5                               # abrupt KMP exit, no successor named
        else:
            p -= 20                                  # independent director / CS shuffle
    if in_portfolio:
        p += 10
    return min(p, 100)


# --------------------------------------------------------------------------- #
# 6. Digest builder
# --------------------------------------------------------------------------- #
def build_digest(filings: list[Filing], now: datetime | None = None,
                 window_hours: int = 24, portfolio: set[str] | None = None,
                 min_priority: int = 40) -> dict:
    now = now or datetime.now(IST)
    portfolio = PORTFOLIO_TICKERS if portfolio is None else portfolio
    start = now - timedelta(hours=window_hours)

    stats = {"received": len(filings), "outside_window": 0, "routine_dropped": 0,
             "below_threshold": 0, "merged_duplicates": 0}
    best: dict[tuple[str, str], DigestItem] = {}

    for f in sorted(filings, key=lambda x: x.filed_at):
        if not (start <= f.filed_at <= now):
            stats["outside_window"] += 1
            continue
        cat = classify(f)
        if cat == "ROUTINE":
            stats["routine_dropped"] += 1
            continue

        details: dict = {}
        if cat == "RESULTS":
            details = extract_results(f.body)
            headline = results_headline(f.ticker, details)
        elif cat == "MGMT_CHANGE":
            details = extract_mgmt_change(f.body)
            if details["person"]:
                details["bio"] = mock_linkedin_bio(details["person"], " & ".join(details["all_roles"]) or None, f.company)
            headline = mgmt_headline(f.ticker, details)
        else:
            headline = f"{f.ticker.split('.')[0]}: {f.subject}"[:120]

        held = f.ticker in portfolio
        item = DigestItem(f.id, f.ticker, f.company, f.filed_at.isoformat(), cat,
                          score(cat, details, held), headline, f.subject, held, details)

        if item.priority < min_priority:
            stats["below_threshold"] += 1
            continue

        # Dedupe: results PDF + press release + investor deck -> one item.
        # Management changes are keyed per person so two exits don't collapse.
        key = (f.ticker, cat if cat != "MGMT_CHANGE" else f"{cat}:{details.get('person')}")
        if key in best:
            stats["merged_duplicates"] += 1
            keep, other = (best[key], item)
            richer = len(item.details.get("metrics", {})) > len(keep.details.get("metrics", {}))
            if item.priority > keep.priority or richer:
                keep, other = item, keep
            keep.merged_filing_ids += [other.filing_id] + other.merged_filing_ids
            best[key] = keep
        else:
            best[key] = item

    items = sorted(best.values(), key=lambda i: (-i.priority, i.filed_at))
    return {
        "generated_at": now.isoformat(),
        "window": {"from": start.isoformat(), "to": now.isoformat(), "hours": window_hours},
        "stats": stats,
        "items": [asdict(i) for i in items],
    }


# --------------------------------------------------------------------------- #
# 7. Mock filing generator (swap for an NSE/BSE announcements fetcher later)
# --------------------------------------------------------------------------- #
def _results_body(period: str, rows: dict[str, tuple]) -> str:
    lines = [f"Statement of Standalone Unaudited Financial Results for the quarter ended {period}",
             "(₹ in crore)",
             "Particulars | Quarter ended (current) | Quarter ended (preceding) | Quarter ended (corresponding PY)"]
    for label, vals in rows.items():
        lines.append(f"{label} | " + " | ".join(
            f"({abs(v):,.2f})" if v < 0 else f"{v:,.2f}" for v in vals))
    return "\n".join(lines)


def generate_mock_filings(now: datetime | None = None) -> list[Filing]:
    now = now or datetime.now(IST)
    h = lambda hrs: now - timedelta(hours=hrs)
    name = lambda t: UNIVERSE.get(t, {}).get("name", t.split(".")[0])
    raw = [
        # ---- material ----
        ("INFY.NS", h(3), "Financial Results for the quarter ended 30.09.2026",
         _results_body("30.09.2026", {
             "Revenue from Operations": (41764, 40986, 38994),
             "EBITDA": (10120, 9875, 9310),
             "Net Profit for the period": (7120, 6885, 6403),
             "Earnings per share (Basic)": (17.15, 16.58, 15.42)})),
        ("INFY.NS", h(2.5), "Press Release - Q2 FY27 results",
         "Revenue from Operations | 41,764 | 40,986 | 38,994"),               # duplicate -> merged
        ("TATASTEEL.NS", h(6), "Outcome of Board Meeting - Unaudited Financial Results",
         _results_body("30.09.2026", {
             "Revenue from Operations": (54210, 55870, 53105),
             "Profit after tax": (1840, (-420), (-610))})),                      # loss -> profit
        ("LT.NS", h(9), "Acquisition of 100% stake in a green hydrogen electrolyser company",
         "The Board has approved acquisition of 100% equity stake for a cash consideration of ₹1,250 crore."),
        ("HDFCBANK.NS", h(11), "Resignation of Chief Financial Officer",
         "Mr. Arvind Mehra has tendered his resignation as Chief Financial Officer and KMP of the Bank "
         "due to personal reasons, with effect from December 31, 2026."),
        ("TITAN.NS", h(14), "Appointment of Managing Director & Chief Executive Officer",
         "The Board approved the appointment of Ms. Kavya Raghunathan as Managing Director and Chief Executive "
         "Officer for a term of five years, effective from January 1, 2027, subject to shareholder approval."),
        ("ITC.NS", h(15), "Appointment of Independent Director",
         "Dr. Suresh Nair has been appointed as an Additional Director in the category of Independent Director."),
        ("ADANIPORTS.NS", h(17), "Letter of award received for terminal development",
         "The company has received a letter of award for a container terminal contract valued at ₹3,400 crore."),
        ("SBIN.NS", h(19), "Interim dividend - record date",
         "The Board declared an interim dividend of ₹6 per share. Record date: 28 November 2026."),
        ("SUNPHARMA.NS", h(21), "Receipt of order from US FDA - warning letter",
         "The company received a penalty notice / warning letter order from regulator for its Halol facility."),
        ("BAJFINANCE.NS", h(30), "Financial Results for the quarter ended 30.09.2026",   # outside 24h
         _results_body("30.09.2026", {"Net Interest Income": (9800, 9410, 8350)})),
        # ---- routine (dropped) ----
        ("RELIANCE.NS", h(4), "Compliance Certificate under Regulation 74(5) of SEBI (DP) Regulations, 2018",
         "Certificate from RTA confirming dematerialisation requests processed."),
        ("TCS.NS", h(5), "Closure of Trading Window", "Trading window shall remain closed from Jan 1, 2027."),
        ("ICICIBANK.NS", h(7), "Intimation of loss of share certificates", "Folio no. ...; duplicate certificates to be issued."),
        ("MARUTI.NS", h(8), "Newspaper Publication", "Copies of newspaper advertisement regarding postal ballot notice."),
        ("NTPC.NS", h(12), "Intimation of Analyst Meet", "The company will participate in an investor conference."),
        ("ULTRACEMCO.NS", h(20), "Certificate under Regulation 7(3) of SEBI LODR", "Certificate signed by Company Secretary."),
    ]
    return [Filing(f"MOCK-{i:04d}", t, name(t), ts, s, b) for i, (t, ts, s, b) in enumerate(raw, 1)]


# --------------------------------------------------------------------------- #
# 8. CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="FINN Push-mode filings digest")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--min-priority", type=int, default=40)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    now = datetime.now(IST)
    digest = build_digest(generate_mock_filings(now), now, args.hours, min_priority=args.min_priority)

    s = digest["stats"]
    print(f"[FINN Push] window {args.hours}h | received {s['received']} | outside window {s['outside_window']} "
          f"| routine dropped {s['routine_dropped']} | below threshold {s['below_threshold']} "
          f"| merged {s['merged_duplicates']}\n")
    for it in digest["items"]:
        flag = "★" if it["in_portfolio"] else " "
        print(f"{flag} [{it['priority']:>3}] {it['category']:<11} {it['headline']}")
        bio = it["details"].get("bio")
        if bio:
            print(f"      └ (mock bio) {bio['headline']} — {bio['snippet']}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    path = out / f"push_digest_{now:%Y-%m-%d_%H%M}.json"
    path.write_text(json.dumps(digest, indent=2, ensure_ascii=False))
    print(f"\n[FINN Push] saved {path}")


if __name__ == "__main__":
    main()
