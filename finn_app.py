"""
FINN Terminal - mobile-first Streamlit UI over the three backend modules.

    pip install streamlit yfinance pandas
    streamlit run finn_app.py

Keep finn_data.py, finn_push.py and finn_pull.py in the same folder.
Set FINN_OFFLINE=1 to skip Yahoo and use simulated prices (badged SIM in the UI).
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import finn_data as fd
import finn_push as fp
import finn_pull as fpl

IST = fp.IST
st.set_page_config(page_title="FINN Terminal", page_icon="▮", layout="wide",
                   initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
:root{
  --bg:#07090a; --panel:#0c1011; --panel2:#101617; --line:#1b2427; --line2:#253236;
  --tx:#c8d2d4; --dim:#6a7a7e; --mute:#45534f; --amb:#f5a524; --cy:#4cc9f0;
  --up:#00d26a; --dn:#ff4d4f; --upbg:rgba(0,210,106,.10); --dnbg:rgba(255,77,79,.10);
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{background:var(--bg)!important;color:var(--tx);font-family:var(--mono)!important}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stAppDeployButton"],#MainMenu,footer,[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important}
.block-container{padding:0 10px 60px!important;max-width:1180px!important}
[data-testid="stVerticalBlock"]{gap:.45rem!important}
*:not([data-testid='stIconMaterial']):not(.material-symbols-rounded){font-family:var(--mono)!important}
p,span,div,label{font-size:12px}

/* --- streamlit widgets restyled --- */
button[data-variant="segmented_control"],button[data-variant="pills"]{
  background:var(--panel)!important;border:1px solid var(--line2)!important;border-radius:2px!important;
  color:var(--dim)!important;letter-spacing:.08em;min-height:28px!important;padding:2px 12px!important}
button[data-variant="segmented_control"] p,button[data-variant="pills"] p{font-size:11px!important;color:inherit!important}
button[data-variant="segmented_control"][aria-checked="true"],button[data-variant="pills"][aria-checked="true"]{
  background:var(--amb)!important;color:#000!important;border-color:var(--amb)!important;font-weight:700}
button[data-variant="segmented_control"]:hover,button[data-variant="pills"]:hover{color:var(--amb)!important;border-color:var(--amb)!important}
button[data-variant="segmented_control"][aria-checked="true"]:hover{color:#000!important}
[role="radiogroup"]{flex-wrap:wrap!important;gap:4px!important}
[data-testid="stWidgetLabel"] p{color:var(--mute)!important;font-size:10px!important;letter-spacing:.12em;text-transform:uppercase}
[data-testid="stFileUploaderDropzone"]{background:var(--panel)!important;border:1px dashed var(--line2)!important;border-radius:2px}
.stSelectbox div[data-baseweb="select"]>div{background:var(--panel)!important;border-color:var(--line2)!important;border-radius:2px;min-height:30px}

/* --- top bar & tape --- */
.top{position:sticky;top:0;z-index:50;background:var(--bg);border-bottom:1px solid var(--line);
  display:flex;align-items:center;gap:12px;padding:8px 2px;flex-wrap:wrap}
.logo{color:var(--amb);font-weight:700;font-size:15px;letter-spacing:.18em}
.logo i{display:inline-block;width:8px;height:14px;background:var(--amb);margin-left:3px;vertical-align:-2px;animation:blink 1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
.idx{font-size:12px}.idx b{color:#fff;font-weight:500}
.sp{flex:1}.meta{color:var(--dim);font-size:10px;letter-spacing:.08em}
.badge{font-size:9px;padding:1px 5px;border:1px solid;border-radius:2px;letter-spacing:.1em}
.b-live{color:var(--up);border-color:var(--up)}.b-sim{color:var(--amb);border-color:var(--amb)}
.b-mock{color:var(--cy);border-color:var(--cy)}
.tape{overflow:hidden;white-space:nowrap;border-bottom:1px solid var(--line);padding:5px 0;mask-image:linear-gradient(90deg,transparent,#000 4%,#000 96%,transparent)}
.tape-in{display:inline-block;animation:tape 70s linear infinite}
.tape span{margin-right:22px;font-size:11px}.tape em{font-style:normal;color:var(--dim);margin-right:6px}
@keyframes tape{from{transform:translateX(0)}to{transform:translateX(-50%)}}

.up{color:var(--up)!important}.dn{color:var(--dn)!important}.fl{color:var(--dim)!important}
.amb{color:var(--amb)}.cy{color:var(--cy)}.dim{color:var(--dim)}.w{color:#fff}

/* --- panels --- */
.sec{border:1px solid var(--line);background:var(--panel);margin:0 0 8px}
.sec>summary,.hd{list-style:none;cursor:pointer;padding:7px 10px;border-bottom:1px solid var(--line);
  font-size:10px;letter-spacing:.14em;color:var(--amb);text-transform:uppercase;display:flex;gap:8px;align-items:center}
.sec>summary::-webkit-details-marker{display:none}
.sec>summary:before{content:'▸';color:var(--mute)}.sec[open]>summary:before{content:'▾'}
.sec>summary .r{margin-left:auto;color:var(--dim);letter-spacing:.04em;text-transform:none}
.bd{padding:8px 10px}
.statline{font-size:10px;color:var(--dim);letter-spacing:.06em;padding:6px 0 2px}
.statline b{color:var(--tx);font-weight:500}

/* key-value grid */
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));border-top:1px solid var(--line);border-left:1px solid var(--line)}
.kv>div{background:var(--panel);padding:6px 8px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.kv .k{font-size:9px;color:var(--dim);letter-spacing:.1em;text-transform:uppercase}
.kv .v{font-size:14px;color:#fff;margin-top:2px}.kv .p{font-size:10px;color:var(--dim)}
.flag{font-size:11px;padding:4px 8px;margin-top:6px;border-left:2px solid var(--dn);background:var(--dnbg);color:#ffb3b3}
.ok{font-size:11px;padding:4px 8px;margin-top:6px;border-left:2px solid var(--up);background:var(--upbg);color:#a8f0c8}
.src{margin-top:6px}.src summary{font-size:10px;color:var(--cy);cursor:pointer;list-style:none}
.src summary::-webkit-details-marker{display:none}
.src .q{font-size:10.5px;color:var(--dim);border-left:1px solid var(--line2);padding:3px 8px;margin:4px 0}
.src .q b{color:var(--tx);font-weight:500}

/* tables */
.tb{width:100%;border-collapse:collapse;font-size:11.5px}
.tb th{font-size:9px;color:var(--dim);letter-spacing:.1em;text-transform:uppercase;font-weight:400;text-align:right;padding:5px 6px;border-bottom:1px solid var(--line2);white-space:nowrap}
.tb td{padding:5px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
.tb th:first-child,.tb td:first-child{text-align:left}
.tb tr:hover td{background:var(--panel2)}
.tbw{overflow-x:auto}
.tk{color:#fff;font-weight:500}.sub{display:block;font-size:9px;color:var(--dim)}
.chip{font-size:9px;padding:1px 5px;border-radius:2px;letter-spacing:.06em}
.c-up{background:var(--upbg);color:var(--up)}.c-dn{background:var(--dnbg);color:var(--dn)}
.c-fl{background:var(--panel2);color:var(--dim)}.c-amb{background:rgba(245,165,36,.12);color:var(--amb)}

/* push feed */
.card{border:1px solid var(--line);border-left:3px solid var(--mute);background:var(--panel);margin-bottom:6px}
.card.p-hot{border-left-color:var(--dn)}.card.p-hi{border-left-color:var(--amb)}.card.p-md{border-left-color:var(--cy)}
.card>summary{list-style:none;cursor:pointer;padding:8px 10px}
.card>summary::-webkit-details-marker{display:none}
.row1{display:flex;gap:8px;align-items:center;font-size:10px;color:var(--dim)}
.pr{font-weight:700;color:#000;background:var(--mute);padding:0 4px;font-size:10px}
.p-hot .pr{background:var(--dn)}.p-hi .pr{background:var(--amb)}.p-md .pr{background:var(--cy)}
.row1 .tk{font-size:12px}.held{color:var(--amb)}
.hl{font-size:12.5px;color:var(--tx);margin-top:4px;line-height:1.45}
.card .bd{border-top:1px dashed var(--line2)}
.bio{border:1px solid var(--line2);padding:6px 8px;margin-top:6px;background:var(--panel2)}
.bio .t{color:#fff;font-size:11.5px}.bio .s{color:var(--dim);font-size:11px;margin-top:3px;line-height:1.45}

/* pull: verdict strip + terminal block */
.hit{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.hit .n{font-size:30px;color:#fff;font-weight:700}
.strip{display:flex;gap:3px;flex-wrap:wrap;margin-top:4px}
.strip i{width:14px;height:14px;display:inline-block}
.v-met,.v-beat{background:var(--up)}.v-beat{box-shadow:inset 0 0 0 2px #000}.v-missed{background:var(--dn)}
.v-other{background:var(--mute)}
.term{background:#000;border:1px solid var(--line2);padding:8px 10px;font-size:11px;line-height:1.6;color:#b9c4c6;white-space:pre-wrap}
.term .ps{color:var(--up)}

/* book */
.big{font-size:22px;color:#fff;font-weight:500}
.bar{height:4px;background:var(--line);margin-top:3px}.bar>i{display:block;height:100%;background:var(--cy)}
.grid2{display:grid;grid-template-columns:1fr;gap:8px}
@media(min-width:900px){.grid2{grid-template-columns:1fr 1fr}.hide-d{display:none}}
@media(max-width:600px){.hide-m{display:none}.kv{grid-template-columns:repeat(2,1fr)}.tb{font-size:11px}}
</style>
"""
st.html(CSS)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
esc = lambda s: html.escape(str(s)) if s is not None else ""
sym = lambda t: t.split(".")[0].lstrip("^")


def cls(v):
    return "fl" if v is None or (isinstance(v, float) and np.isnan(v)) or v == 0 else ("up" if v > 0 else "dn")


def pct(v, d=2, sign=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.{d}f}%" if sign else f"{v:.{d}f}%"


def inr(v, d=0):
    if v is None:
        return "—"
    a = abs(v)
    s = f"{a/1e7:,.2f}Cr" if a >= 1e7 else f"{a/1e5:,.2f}L" if a >= 1e5 else f"{a:,.{d}f}"
    return ("-" if v < 0 else "") + "₹" + s


def ago(iso: str) -> str:
    m = int((datetime.now(IST) - datetime.fromisoformat(iso)).total_seconds() // 60)
    return f"{m}m" if m < 60 else f"{m//60}h{m%60:02d}"


def spark(vals, w=70, h=18) -> str:
    """Inline sparkline as a data-URI SVG <img> (st.html strips raw <svg>)."""
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1
    pts = " ".join(f"{i*(w/(len(vals)-1)):.1f},{h-(v-lo)/rng*(h-2)-1:.1f}" for i, v in enumerate(vals))
    col = "#00d26a" if vals[-1] >= vals[0] else "#ff4d4f"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
           f'<polyline fill="none" stroke="{col}" stroke-width="1.2" points="{pts}"/></svg>')
    return f'<img alt="" width="{w}" height="{h}" style="vertical-align:middle" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}">'


def sources_block(items: list[dict]) -> str:
    qs = [e for e in items if e]
    if not qs:
        return ""
    rows = "".join(
        f'<div class="q"><b>{esc(e.get("doc_id"))}</b> · {esc(e.get("period"))}'
        f'{" · p." + str(e["page"]) if e.get("page") else ""}'
        f'{" · " + esc(e["speaker"]) if e.get("speaker") else ""}<br>“{esc(e.get("quote"))}”</div>' for e in qs)
    return f'<details class="src"><summary>⌕ {len(qs)} primary source{"s" * (len(qs) > 1)}</summary>{rows}</details>'


# --------------------------------------------------------------------------- #
# Data loaders (cached)
# --------------------------------------------------------------------------- #
def _synthetic(tickers: list[str], days=260) -> pd.DataFrame:
    held = {h.ticker: h.avg_cost for h in fd.MOCK_PORTFOLIO}
    dates = pd.bdate_range(end=datetime.now(IST).date() - timedelta(days=1), periods=days)
    frames = []
    for t in tickers:
        rng = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
        base = 24000 if t == fd.BENCHMARK else held.get(t, rng.uniform(200, 4000))
        drift = rng.normal(0.0004, 0.0004)
        path = np.exp(np.cumsum(rng.normal(drift, 0.014, days)))
        close = base * rng.uniform(0.8, 1.05) * path / path[0]
        vol = rng.integers(2e5, 5e6) * np.exp(rng.normal(0, .35, days))
        vol[-1] *= rng.choice([1, 1, 1, 2.2])        # occasional volume spike
        frames.append(pd.DataFrame({"Date": dates, "Ticker": t, "Open": close, "High": close * 1.008,
                                    "Low": close * .992, "Close": close, "Volume": vol.astype(int)}))
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=900, show_spinner=False)
def load_market(period="1y"):
    tickers = list(fd.UNIVERSE) + [fd.BENCHMARK]
    source = "LIVE"
    if os.getenv("FINN_OFFLINE") == "1":
        raw, source = _synthetic(tickers), "SIM"
    else:
        try:
            raw = fd.fetch_eod(tickers, period=period, retries=1)
        except Exception:
            raw, source = _synthetic(tickers), "SIM"
    df = fd.add_indicators(raw)
    snap = fd.latest_snapshot(df)
    pf, summary = fd.value_portfolio(snap, fd.MOCK_PORTFOLIO)
    return df, snap, pf, summary, source


@st.cache_data(ttl=300, show_spinner=False)
def load_push(hours: int, _bucket: str):
    now = datetime.now(IST)
    return fp.build_digest(fp.generate_mock_filings(now), now, hours)


@st.cache_data(show_spinner=False)
def load_pull(jsonl: str | None):
    if jsonl:
        chunks = [fpl.Chunk(**json.loads(l)) for l in jsonl.splitlines() if l.strip()]
    else:
        chunks = fpl.mock_chunks()
    return fpl.build_research(chunks)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
df, snap, pf, summary, source = load_market()
now = datetime.now(IST)
mkt_open = now.weekday() < 5 and (now.hour, now.minute) >= (9, 15) and (now.hour, now.minute) <= (15, 30)
nifty = snap.loc[fd.BENCHMARK] if fd.BENCHMARK in snap.index else None
nifty_html = (f'<span class="idx">NIFTY <b>{nifty.Close:,.2f}</b> '
              f'<span class="{cls(nifty.Ret_1D)}">{pct(nifty.Ret_1D*100)}</span></span>' if nifty is not None else "")
st.html(f"""
<div class="top"><span class="logo">FINN<i></i></span>{nifty_html}<span class="sp"></span>
<span class="meta">{'● MKT OPEN' if mkt_open else '○ MKT CLOSED'} · {now:%d %b %H:%M} IST</span>
<span class="badge {'b-live' if source == 'LIVE' else 'b-sim'}">{source}</span></div>""")

tape = "".join(f'<span><em>{sym(t)}</em>{r.Close:,.2f} <b class="{cls(r.Ret_1D)}">{pct(r.Ret_1D*100)}</b></span>'
               for t, r in snap.drop(index=fd.BENCHMARK, errors="ignore").iterrows())
st.html(f'<div class="tape"><div class="tape-in">{tape}{tape}</div></div>')

mode = st.segmented_control("mode", ["PUSH", "PULL", "BOOK"], default="PUSH",
                            label_visibility="collapsed", key="mode") or "PUSH"


# --------------------------------------------------------------------------- #
# PUSH
# --------------------------------------------------------------------------- #
def render_push():
    c1, c2 = st.columns([3, 1])
    with c1:
        filt = st.pills("filter", ["ALL", "HELD", "RESULTS", "M&A", "MGMT", "OTHER"], default="ALL",
                        label_visibility="collapsed", key="pf") or "ALL"
    with c2:
        hours = st.segmented_control("window", [24, 48, 72], default=24, format_func=lambda h: f"{h}H",
                                     label_visibility="collapsed", key="win") or 24
    d = load_push(hours, now.strftime("%Y%m%d%H%M")[:-1])
    s = d["stats"]
    items = d["items"]
    if filt == "HELD":
        items = [i for i in items if i["in_portfolio"]]
    elif filt == "MGMT":
        items = [i for i in items if i["category"] == "MGMT_CHANGE"]
    elif filt == "OTHER":
        items = [i for i in items if i["category"] not in ("RESULTS", "M&A", "MGMT_CHANGE")]
    elif filt != "ALL":
        items = [i for i in items if i["category"] == filt]

    st.html(f'<div class="statline">{hours}H WINDOW ▸ <b>{s["received"]}</b> FILINGS · '
            f'<b>{s["routine_dropped"]}</b> ROUTINE DROPPED · <b>{s["merged_duplicates"]}</b> MERGED · '
            f'<b>{len(items)}</b> SHOWN <span class="badge b-mock">MOCK FEED</span></div>')

    cards = []
    for it in items:
        p = it["priority"]
        tier = "p-hot" if p >= 90 else "p-hi" if p >= 70 else "p-md" if p >= 55 else ""
        det = it["details"]
        body = ""
        if it["category"] == "RESULTS" and det.get("metrics"):
            rows = "".join(
                f'<tr><td>{esc(k.upper())}</td><td class="w">{m["current"]:,.2f}</td>'
                f'<td class="{cls(m["qoq_pct"])}">{pct(m["qoq_pct"], 1) if m["qoq_pct"] is not None else ("vs loss" if (m["prev_quarter"] or 0) < 0 else "—")}</td>'
                f'<td class="{cls(m["yoy_pct"]) if m["yoy_pct"] is not None else "up" if m.get("note") == "turned profitable YoY" else "dn" if m.get("note") else "fl"}">'
                f'{pct(m["yoy_pct"], 1) if m["yoy_pct"] is not None else esc(m.get("note", "—"))}</td></tr>'
                for k, m in det["metrics"].items())
            body = (f'<div class="dim" style="font-size:10px">PERIOD {esc(det.get("period_end"))} · {esc(det.get("unit"))}</div>'
                    f'<div class="tbw"><table class="tb"><tr><th>Metric</th><th>Current</th><th>QoQ</th><th>YoY</th></tr>{rows}</table></div>')
        elif it["category"] == "MGMT_CHANGE":
            kv = [("Person", det.get("person")), ("Event", (det.get("type") or "").upper()),
                  ("Role", " & ".join(det.get("all_roles") or []) or "—"), ("Effective", det.get("effective") or "—"),
                  ("Reason", det.get("reason") or "—"), ("KMP", "YES" if det.get("is_kmp") else "NO")]
            body = '<div class="kv">' + "".join(f'<div><div class="k">{k}</div><div class="v" style="font-size:12px">{esc(v)}</div></div>' for k, v in kv) + "</div>"
            if (b := det.get("bio")):
                body += (f'<div class="bio"><span class="badge b-mock">MOCK BIO</span> '
                         f'<span class="t">{esc(b["headline"])}</span><div class="s">{esc(b["snippet"])}</div></div>')
        else:
            body = f'<div class="dim">{esc(it["subject"])}</div>'
        merged = f' · merged {", ".join(it["merged_filing_ids"])}' if it["merged_filing_ids"] else ""
        body += f'<div class="dim" style="font-size:9.5px;margin-top:6px">{esc(it["filing_id"])}{merged} · {esc(it["company"])}</div>'
        cat = it["category"].replace("_CHANGE", "").replace("_", " ")
        cards.append(
            f'<details class="card {tier}"><summary><div class="row1"><span class="pr">{p}</span>'
            f'<span class="tk">{sym(it["ticker"])}</span><span class="chip c-amb">{esc(cat)}</span>'
            f'{"<span class=held>★ HELD</span>" if it["in_portfolio"] else ""}<span class="sp"></span>'
            f'<span>{ago(it["filed_at"])} ago</span></div><div class="hl">{esc(it["headline"])}</div></summary>'
            f'<div class="bd">{body}</div></details>')
    st.html("".join(cards) or '<div class="dim" style="padding:20px 0">No material filings in window.</div>')


# --------------------------------------------------------------------------- #
# PULL
# --------------------------------------------------------------------------- #
def kv_grid(cat: dict, keys: list[tuple[str, str]]) -> tuple[str, list]:
    cells, ev = [], []
    for key, label in keys:
        f = cat.get(key)
        if not isinstance(f, dict) or f.get("value") is None:
            continue
        v, u, pr = f["value"], f.get("unit") or "", f.get("prior")
        fmt = (lambda x: f"₹{x:,.0f} cr") if u == "₹ cr" else (lambda x: f"{x}{'' if u in ('', None) else (' ' + u if u not in ('%', 'x') else u)}")
        delta = ""
        if isinstance(v, (int, float)) and isinstance(pr, (int, float)) and pr:
            ch = (v / pr - 1) * 100 if u not in ("%", "x", "days") else v - pr
            delta = f' <span class="{cls(ch)}">{ch:+.1f}{"%" if u not in ("%", "x", "days") else ("pp" if u == "%" else "")}</span>'
        prior = f'<div class="p">PY {fmt(pr)}{delta}</div>' if isinstance(pr, (int, float)) else ""
        cells.append(f'<div><div class="k">{esc(label)}</div><div class="v">{esc(fmt(v)) if isinstance(v, (int, float)) else esc(v)}</div>{prior}</div>')
        if f.get("evidence"):
            ev.append(f["evidence"])
    return '<div class="kv">' + "".join(cells) + "</div>", ev


def flags_html(flags: list[str], ok_msg: str) -> str:
    return "".join(f'<div class="flag">! {esc(x)}</div>' for x in flags) if flags else f'<div class="ok">✓ {ok_msg}</div>'


def section(title: str, inner: str, right: str = "", open_=True) -> str:
    return (f'<details class="sec" {"open" if open_ else ""}><summary>{esc(title)}'
            f'<span class="r">{right}</span></summary><div class="bd">{inner}</div></details>')


def render_pull():
    c1, c2 = st.columns([2, 1])
    with c2:
        up = st.file_uploader("Load chunks (.jsonl)", type=["jsonl"], label_visibility="collapsed")
    res = load_pull(up.getvalue().decode() if up else None)
    C = res["categories"]
    with c1:
        st.html(f'<div style="padding-top:4px"><span class="w" style="font-size:15px">{esc(res["company"])}</span>'
                f'<div class="statline">AR <b>{esc(res["annual_report_period"])}</b> · <b>{len(res["sources_used"])}</b> PRIMARY DOCS · '
                f'<b>{len(res["rejected_sources"])}</b> REJECTED (NON-PRIMARY) · MISSING: '
                f'<b>{esc(", ".join(res["missing_fields"]) or "none")}</b></div></div>')

    left, right = [], []

    # 1. business
    b = C["business_snapshot"] or {}
    inner = "".join(f'<div style="font-size:12px;line-height:1.55;margin-bottom:4px">{esc(b[k]["value"])}</div>'
                    for k in ("description", "segments") if b.get(k))
    inner += sources_block([b[k].get("evidence") for k in ("description", "segments") if b.get(k)])
    left.append(section("01 Business snapshot", inner))

    # 2. financials
    g, ev = kv_grid(C["financial_snapshot"], [("revenue", "Revenue"), ("ebitda", "EBITDA"), ("pat", "PAT"),
                                               ("ebitda_margin_pct", "EBITDA mgn"), ("pat_margin_pct", "PAT mgn"),
                                               ("revenue_growth_pct", "Rev growth"), ("pat_growth_pct", "PAT growth")])
    left.append(section("02 Financial snapshot", g + sources_block(ev), res["annual_report_period"]))

    # 3. debt
    d = C["debt_profile"]
    g, ev = kv_grid(d, [("gross_debt", "Gross debt"), ("cash", "Cash"), ("net_debt", "Net debt"),
                        ("net_debt_to_ebitda", "ND/EBITDA"), ("debt_to_equity", "D/E"),
                        ("interest_coverage", "Int. cover"), ("equity", "Net worth")])
    left.append(section("03 Debt profile", g + flags_html(d["flags"], "Leverage within limits") + sources_block(ev),
                        f'{len(d["flags"])} flag(s)'))

    # 4. cash
    cq = C["cash_flow_quality"]
    g, ev = kv_grid(cq, [("cfo", "CFO"), ("capex", "Capex"), ("free_cash_flow", "FCF"),
                         ("cfo_to_ebitda", "CFO/EBITDA"), ("cfo_to_pat", "CFO/PAT"), ("receivable_days", "Debtor days")])
    left.append(section("04 Cash-flow quality", g + flags_html(cq["flags"], "Cash conversion healthy") + sources_block(ev),
                        f'{len(cq["flags"])} flag(s)'))

    # 5. capital allocation
    ca = C["capital_allocation"]
    g, ev = kv_grid(ca, [("capex", "Capex"), ("capex_to_revenue_pct", "Capex/Rev"), ("dividend_per_share", "DPS"),
                         ("dividend_total", "Div payout"), ("dividend_payout_pct", "Payout ratio")])
    if ca.get("acquisitions"):
        g += f'<div style="margin-top:6px;font-size:11.5px"><span class="chip c-amb">M&A</span> {esc(ca["acquisitions"]["value"])}</div>'
        ev.append(ca["acquisitions"].get("evidence"))
    left.append(section("05 Capital allocation", g + sources_block(ev)))

    # 7. guidance vs delivery (headline panel on the right column)
    gv = C["guidance_vs_delivery"]
    sc = gv["scorecard"]
    strip = "".join(f'<i class="v-{r["verdict"] if r["verdict"] in ("met", "beat", "missed") else "other"}" '
                    f'title="{esc(r["period"])} {esc(r["metric"])}: {esc(r["verdict"])}"></i>' for r in gv["records"])
    rows = ""
    for r in gv["records"]:
        u = "%" if r["unit"] == "%" else " cr"
        lo, hi = r["guided"]
        guided = f"≥{lo}{u}" if hi is None else (f"~{lo}{u}" if lo == hi else f"{lo}–{hi}{u}")
        vc = {"met": "c-up", "beat": "c-up", "missed": "c-dn"}.get(r["verdict"], "c-fl")
        xc = ' <span class="cy" title="matches audited annual report">✓AR</span>' if r.get("annual_report_cross_check", {}).get("matches") else ""
        rows += (f'<tr><td><span class="tk">{esc(r["period"])}</span><span class="sub">{esc(r["metric"].replace("_", " "))}</span></td>'
                 f'<td>{guided}<span class="sub">{esc(r["guided_in"])}</span></td><td class="w">{r["actual"]}{u}{xc}</td>'
                 f'<td class="{cls(r["deviation"])}">{f"{r['deviation']:+}" if r["deviation"] else "·"}</td><td><span class="chip {vc}">{esc(r["verdict"].upper())}</span></td></tr>')
    inner = (f'<div class="hit"><span class="n">{sc["hit_rate_pct"] if sc["hit_rate_pct"] is not None else "—"}%</span>'
             f'<span class="dim">HIT RATE · {sc["met_or_beat"]}/{sc["scored"]} met or beat · {esc(", ".join(gv["window_quarters"]))}</span></div>'
             f'<div class="strip">{strip}</div>'
             f'<div class="tbw" style="margin-top:8px"><table class="tb"><tr><th>Period</th><th>Guided</th><th>Actual</th><th>Δ</th><th>Verdict</th></tr>{rows}</table></div>')
    if (we := gv.get("worked_example")):
        steps = "\n".join(f'<span class="ps">$</span> {esc(s)}' for s in we["steps"])
        inner += f'<div class="hd" style="border:0;padding:10px 0 4px">▸ Worked example · {esc(we["period"])} {esc(we["metric"])}</div><div class="term">{steps}</div>'
    inner += sources_block([e for r in gv["records"] for e in (r["guidance_evidence"], r["actual_evidence"])])
    right.append(section("07 Guidance vs delivery", inner, "last 4Q"))

    # 6. guidance outstanding + revisions
    mg = C["management_guidance"]
    rows = "".join(
        f'<tr><td><span class="tk">{esc(g_["target_period"])}</span><span class="sub">{esc(g_["metric"].replace("_", " "))}</span></td>'
        f'<td class="w">{g_["low"]}{"" if g_["high"] is None else "–" + str(g_["high"])}{"%" if g_["unit"] == "%" else " cr"}</td>'
        f'<td>{esc(g_["given_in"])}{" <span class=amb>*inferred</span>" if g_["target_inferred"] else ""}</td></tr>'
        for g_ in mg["outstanding"])
    revs = "".join(
        f'<div class="{"flag" if r["direction"] == "cut" else "ok"}">{"▼" if r["direction"] == "cut" else "▲"} '
        f'{esc(r["metric"].replace("_", " "))} {esc(r["target_period"])}: {r["from"]} ({esc(r["from_call"])}) → '
        f'{r["to"]} ({esc(r["to_call"])}) — {esc(r["direction"].upper())}</div>' for r in mg["revisions"])
    inner = (f'<div class="tbw"><table class="tb"><tr><th>Target</th><th>Guided</th><th>Given in</th></tr>{rows}</table></div>' if rows
             else '<div class="dim">No outstanding guidance.</div>') + revs
    inner += sources_block([g_["evidence"] for g_ in mg["outstanding"]])
    right.append(section("06 Management guidance", inner, f'{len(mg["revisions"])} revision(s)'))

    # 8. governance
    gov = C["governance_red_flags"]
    g, ev = kv_grid(gov, [("audit_opinion", "Audit opinion"), ("contingent_liabilities", "Contingent liab."),
                          ("contingent_to_networth_pct", "CL / net worth"), ("related_party_transactions", "RPTs"),
                          ("promoter_pledge_pct", "Promoter pledge")])
    ev += [gov[k].get("evidence") for k in ("emphasis_of_matter", "auditor_change") if gov.get(k)]
    right.append(section("08 Governance & red flags", g + flags_html(gov["flags"], "No governance flags") + sources_block(ev),
                         f'{len(gov["flags"])} flag(s)'))

    if res["rejected_sources"]:
        right.append('<div class="dim" style="font-size:10px">REJECTED: ' +
                     ", ".join(f'{esc(r["doc_id"])} ({esc(r["source_type"])})' for r in res["rejected_sources"]) + "</div>")

    st.html(f'<div class="grid2"><div>{"".join(left)}</div><div>{"".join(right)}</div></div>')


# --------------------------------------------------------------------------- #
# BOOK (portfolio + market watch)
# --------------------------------------------------------------------------- #
def render_book():
    closes = df.sort_values("Date").groupby("Ticker")["Close"].apply(lambda s: list(s.tail(30)))
    day_pnl = sum(r.Current - r.Current / (1 + r.Day_Change_Pct / 100) for r in pf.itertuples())
    s = summary
    st.html(f"""
<div class="sec"><div class="hd">Portfolio <span class="r" style="margin-left:auto;color:var(--dim)">{len(pf)} holdings · mock book</span></div>
<div class="kv">
<div><div class="k">Value</div><div class="big">{inr(s['current_value'])}</div></div>
<div><div class="k">Total P&amp;L</div><div class="big {cls(s['pnl'])}">{inr(s['pnl'])}</div><div class="p {cls(s['pnl'])}">{pct(s['pnl_pct'])}</div></div>
<div><div class="k">Day P&amp;L</div><div class="big {cls(day_pnl)}">{inr(day_pnl)}</div></div>
<div><div class="k">Invested</div><div class="big">{inr(s['invested'])}</div></div>
</div></div>""")

    rows = "".join(
        f'<tr><td><span class="tk">{sym(r.Ticker)}</span><span class="sub">{r.Qty} @ {r.Avg_Cost:,.0f}</span></td>'
        f'<td class="hide-m">{spark(closes.get(r.Ticker, []))}</td>'
        f'<td class="w">{r.LTP:,.2f}</td><td class="{cls(r.Day_Change_Pct)}">{pct(r.Day_Change_Pct)}</td>'
        f'<td class="{cls(r.PnL)}">{inr(r.PnL)}<span class="sub {cls(r.PnL)}">{pct(r.PnL_Pct)}</span></td>'
        f'<td class="hide-m">{r.Weight_Pct:.1f}%<div class="bar"><i style="width:{min(r.Weight_Pct * 2.5, 100):.0f}%"></i></div></td></tr>'
        for r in pf.sort_values("Current", ascending=False).itertuples())
    holdings = (f'<div class="tbw"><table class="tb"><tr><th>Holding</th><th class="hide-m">30D</th><th>LTP</th><th>Day</th>'
                f'<th>P&amp;L</th><th class="hide-m">Wt</th></tr>{rows}</table></div>')

    sectors = "".join(
        f'<div style="display:flex;justify-content:space-between;font-size:11px;margin-top:5px"><span>{esc(k)}</span>'
        f'<span class="dim">{v:.1f}%</span></div><div class="bar"><i style="width:{v:.0f}%;background:var(--amb)"></i></div>'
        for k, v in sorted(s["sector_weights_pct"].items(), key=lambda x: -x[1]))

    m = snap.drop(index=fd.BENCHMARK, errors="ignore").sort_values("Ret_1D", ascending=False)
    arrow = lambda b: '<span class="up">▲</span>' if b else '<span class="dn">▼</span>'
    wrows = "".join(
        f'<tr><td><span class="tk">{sym(t)}</span><span class="sub">{esc(r.Sector)}</span></td>'
        f'<td class="w">{r.Close:,.2f}</td><td class="{cls(r.Ret_1D)}">{pct(r.Ret_1D*100)}</td>'
        f'<td class="{cls(r.Ret_5D)} hide-m">{pct(r.Ret_5D*100, 1)}</td>'
        f'<td>{arrow(r.Above_SMA_50)}{arrow(r.Above_SMA_200) if r.SMA_200 == r.SMA_200 else "<span class=dim>·</span>"}</td>'
        f'<td class="{"amb" if r.Vol_Ratio >= 1.5 else "dim"}">{r.Vol_Ratio:.1f}x</td>'
        f'<td class="{cls(r.Pct_From_52W_High)} hide-m">{pct(r.Pct_From_52W_High*100, 1)}</td></tr>'
        for t, r in m.iterrows())
    watch = (f'<div class="tbw"><table class="tb"><tr><th>Symbol</th><th>LTP</th><th>1D</th><th class="hide-m">5D</th>'
             f'<th title="above 50D / 200D SMA">50·200</th><th title="volume vs 20D avg">Vol</th>'
             f'<th class="hide-m">52W Hi</th></tr>{wrows}</table></div>')

    st.html(f'<div class="grid2"><div>{section("Holdings", holdings)}{section("Sector mix", sectors)}</div>'
            f'<div>{section("Market watch", watch, f"{len(m)} symbols · sorted by 1D")}</div></div>')


{"PUSH": render_push, "PULL": render_pull, "BOOK": render_book}[mode]()
st.html('<div class="dim" style="font-size:9.5px;margin-top:14px;border-top:1px solid var(--line);padding-top:6px">'
        'FINN · research tool, not investment advice · filings feed and bios are simulated · '
        f'prices {"via Yahoo Finance (EOD)" if source == "LIVE" else "SIMULATED — Yahoo unreachable"}</div>')
