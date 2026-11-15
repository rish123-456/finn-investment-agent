"""
FINN - Mock portfolio + EOD market data layer for NSE stocks.

Usage:
    pip install yfinance pandas
    python finn_data.py                 # fetch 1y of data, print summary, save outputs
    python finn_data.py --period 2y     # longer history (needed for a reliable SMA-200)

Outputs (in ./data):
    prices_<date>.csv     long-format OHLCV + indicators for every ticker
    snapshot_<date>.json  latest row per ticker + portfolio P&L (what the agent/app reads)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------- #
# 1. Universe: 20 NSE large caps across sectors (Yahoo uses the .NS suffix;
#    use .BO for the BSE listing of the same company).
# --------------------------------------------------------------------------- #
UNIVERSE: dict[str, dict] = {
    "RELIANCE.NS":   {"name": "Reliance Industries",      "sector": "Energy / Conglomerate"},
    "NTPC.NS":       {"name": "NTPC",                     "sector": "Power"},
    "TCS.NS":        {"name": "Tata Consultancy Services","sector": "IT"},
    "INFY.NS":       {"name": "Infosys",                  "sector": "IT"},
    "HDFCBANK.NS":   {"name": "HDFC Bank",                "sector": "Banking"},
    "ICICIBANK.NS":  {"name": "ICICI Bank",               "sector": "Banking"},
    "SBIN.NS":       {"name": "State Bank of India",      "sector": "Banking (PSU)"},
    "BAJFINANCE.NS": {"name": "Bajaj Finance",            "sector": "NBFC"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever",       "sector": "FMCG"},
    "ITC.NS":        {"name": "ITC",                      "sector": "FMCG"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel",            "sector": "Telecom"},
    "LT.NS":         {"name": "Larsen & Toubro",          "sector": "Infrastructure"},
    "ADANIPORTS.NS": {"name": "Adani Ports & SEZ",        "sector": "Logistics"},
    "ULTRACEMCO.NS": {"name": "UltraTech Cement",         "sector": "Cement"},
    "TATASTEEL.NS":  {"name": "Tata Steel",               "sector": "Metals"},
    "MARUTI.NS":     {"name": "Maruti Suzuki",            "sector": "Auto"},
    "SUNPHARMA.NS":  {"name": "Sun Pharmaceutical",       "sector": "Pharma"},
    "DRREDDY.NS":    {"name": "Dr. Reddy's Laboratories", "sector": "Pharma"},
    "ASIANPAINT.NS": {"name": "Asian Paints",             "sector": "Consumer Durables"},
    "TITAN.NS":      {"name": "Titan Company",            "sector": "Consumer Discretionary"},
}

BENCHMARK = "^NSEI"  # NIFTY 50 index, useful for relative performance


# --------------------------------------------------------------------------- #
# 2. Mock portfolio: a subset of the universe with quantities and avg cost.
#    Avg costs are placeholders - swap in whatever your demo user "owns".
# --------------------------------------------------------------------------- #
@dataclass
class Holding:
    ticker: str
    qty: int
    avg_cost: float  # INR per share


MOCK_PORTFOLIO: list[Holding] = [
    Holding("RELIANCE.NS",   20, 1250.0),
    Holding("TCS.NS",        10, 3600.0),
    Holding("HDFCBANK.NS",   25, 1650.0),
    Holding("INFY.NS",       30, 1500.0),
    Holding("ITC.NS",       100,  420.0),
    Holding("SBIN.NS",       40,  780.0),
    Holding("BHARTIARTL.NS", 15, 1600.0),
    Holding("SUNPHARMA.NS",  12, 1700.0),
    Holding("TITAN.NS",       8, 3300.0),
    Holding("TATASTEEL.NS", 150,  145.0),
]


# --------------------------------------------------------------------------- #
# 3. Fetching
# --------------------------------------------------------------------------- #
def fetch_eod(tickers: list[str], period: str = "1y", retries: int = 3) -> pd.DataFrame:
    """Download daily OHLCV for all tickers in one batched call.

    Returns a long DataFrame: Date, Ticker, Open, High, Low, Close, Volume.
    Prices are split/dividend adjusted (auto_adjust=True).
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                tickers,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if raw.empty:
                raise RuntimeError("yfinance returned no data")
            break
        except Exception as e:  # network hiccups / Yahoo rate limits
            last_err = e
            wait = 2 ** attempt
            print(f"[fetch] attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Failed to fetch data after {retries} attempts") from last_err

    frames = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            print(f"[fetch] WARNING: no data for {t}")
            continue
        df = raw[t].dropna(how="all").copy()
        if df.empty:
            print(f"[fetch] WARNING: empty series for {t}")
            continue
        df["Ticker"] = t
        frames.append(df.reset_index())

    out = pd.concat(frames, ignore_index=True)
    return out[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]


# --------------------------------------------------------------------------- #
# 4. Indicators
# --------------------------------------------------------------------------- #
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker moving averages, volume stats, returns and simple signals."""
    df = df.sort_values(["Ticker", "Date"]).copy()
    g = df.groupby("Ticker", group_keys=False)

    # Price moving averages (NaN until enough history exists)
    for w in (20, 50, 200):
        df[f"SMA_{w}"] = g["Close"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    df["EMA_12"] = g["Close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    df["EMA_26"] = g["Close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())

    # Volume
    df["Vol_SMA_20"] = g["Volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["Vol_Ratio"] = df["Volume"] / df["Vol_SMA_20"]  # >1.5 = unusual activity

    # Returns
    df["Ret_1D"] = g["Close"].pct_change(1)
    df["Ret_5D"] = g["Close"].pct_change(5)
    df["Ret_21D"] = g["Close"].pct_change(21)

    # Distance from 52-week high (252 trading days)
    df["High_52W"] = g["High"].transform(lambda s: s.rolling(252, min_periods=1).max())
    df["Pct_From_52W_High"] = df["Close"] / df["High_52W"] - 1

    # Simple, explainable signals for the agent to talk about
    df["Above_SMA_50"] = df["Close"] > df["SMA_50"]
    df["Above_SMA_200"] = df["Close"] > df["SMA_200"]
    df["Golden_Cross"] = df["SMA_50"] > df["SMA_200"]  # trend regime, not the crossover day

    return df


# --------------------------------------------------------------------------- #
# 5. Snapshot + portfolio valuation
# --------------------------------------------------------------------------- #
def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    snap = df.sort_values("Date").groupby("Ticker").tail(1).set_index("Ticker")
    snap["Name"] = snap.index.map(lambda t: UNIVERSE.get(t, {}).get("name", t))
    snap["Sector"] = snap.index.map(lambda t: UNIVERSE.get(t, {}).get("sector", "Index"))
    return snap


def value_portfolio(snap: pd.DataFrame, holdings: list[Holding]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for h in holdings:
        if h.ticker not in snap.index:
            print(f"[portfolio] WARNING: no price for {h.ticker}, skipping")
            continue
        ltp = float(snap.loc[h.ticker, "Close"])
        invested = h.qty * h.avg_cost
        current = h.qty * ltp
        rows.append({
            "Ticker": h.ticker,
            "Name": snap.loc[h.ticker, "Name"],
            "Sector": snap.loc[h.ticker, "Sector"],
            "Qty": h.qty,
            "Avg_Cost": h.avg_cost,
            "LTP": round(ltp, 2),
            "Invested": round(invested, 2),
            "Current": round(current, 2),
            "PnL": round(current - invested, 2),
            "PnL_Pct": round((current / invested - 1) * 100, 2),
            "Day_Change_Pct": round(float(snap.loc[h.ticker, "Ret_1D"]) * 100, 2),
        })
    pf = pd.DataFrame(rows)
    total_inv, total_cur = pf["Invested"].sum(), pf["Current"].sum()
    pf["Weight_Pct"] = (pf["Current"] / total_cur * 100).round(2)

    summary = {
        "invested": round(total_inv, 2),
        "current_value": round(total_cur, 2),
        "pnl": round(total_cur - total_inv, 2),
        "pnl_pct": round((total_cur / total_inv - 1) * 100, 2),
        "sector_weights_pct": pf.groupby("Sector")["Weight_Pct"].sum().round(2).to_dict(),
    }
    return pf, summary


def to_jsonable(snap: pd.DataFrame) -> list[dict]:
    cols = ["Name", "Sector", "Date", "Close", "Volume", "SMA_20", "SMA_50", "SMA_200",
            "EMA_12", "EMA_26", "Vol_Ratio", "Ret_1D", "Ret_5D", "Ret_21D",
            "Pct_From_52W_High", "Above_SMA_50", "Above_SMA_200", "Golden_Cross"]
    recs = []
    for t, row in snap[cols].iterrows():
        rec = {"ticker": t}
        for c in cols:
            v = row[c]
            if isinstance(v, pd.Timestamp):
                v = v.strftime("%Y-%m-%d")
            elif pd.isna(v):
                v = None
            else:
                v = v.item() if hasattr(v, "item") else v  # numpy -> python
                if isinstance(v, float):
                    v = round(v, 4)
            rec[c] = v
        recs.append(rec)
    return recs


# --------------------------------------------------------------------------- #
# 6. Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="FINN EOD data fetcher")
    ap.add_argument("--period", default="1y", help="yfinance period: 6mo, 1y, 2y, 5y ...")
    ap.add_argument("--out", default="data", help="output directory")
    args = ap.parse_args()

    tickers = list(UNIVERSE) + [BENCHMARK]
    print(f"[FINN] Fetching {len(tickers)} symbols, period={args.period} ...")
    df = add_indicators(fetch_eod(tickers, period=args.period))

    snap = latest_snapshot(df)
    pf, summary = value_portfolio(snap, MOCK_PORTFOLIO)

    # ---- console summary ----
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    view = snap.drop(index=BENCHMARK, errors="ignore")[
        ["Close", "SMA_20", "SMA_50", "SMA_200", "Vol_Ratio", "Ret_1D", "Pct_From_52W_High"]
    ].round(2)
    print("\n=== Market snapshot ===")
    print(view.to_string())
    print("\n=== Mock portfolio ===")
    print(pf.to_string(index=False))
    print(f"\nInvested: ₹{summary['invested']:,.0f} | Value: ₹{summary['current_value']:,.0f} "
          f"| P&L: ₹{summary['pnl']:,.0f} ({summary['pnl_pct']}%)")
    if BENCHMARK in snap.index:
        print(f"NIFTY 50: {snap.loc[BENCHMARK, 'Close']:,.2f} "
              f"(1D {snap.loc[BENCHMARK, 'Ret_1D'] * 100:+.2f}%)")

    # ---- persist ----
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = date.today().isoformat()
    df.to_csv(out / f"prices_{tag}.csv", index=False)
    with open(out / f"snapshot_{tag}.json", "w") as f:
        json.dump({
            "as_of": snap["Date"].max().strftime("%Y-%m-%d"),
            "universe": to_jsonable(snap),
            "portfolio": {
                "holdings": pf.to_dict(orient="records"),
                "summary": summary,
            },
        }, f, indent=2)
    print(f"\n[FINN] Saved {out}/prices_{tag}.csv and {out}/snapshot_{tag}.json")


if __name__ == "__main__":
    main()
