"""Turn the trade logs into data/lab_summary.json for the Strategy Lab panel: per-strategy results on both
datasets (raw vs. with the cost/timing fixes), the candidate system's equity curve, and the findings."""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import bars as B
from .analyze import FILTERS
from .engine import INDEX_SYMS, cost_frac
from .strategies import STRATEGIES

OUT = B.ROOT.parent / "lab_summary.json"
TRADABLE = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]      # indices with futures
CANDIDATE = {"strat": "ORB 15m", "trend": "I", "exit": "rr3", "filter": "indices cost<0.2R",
             "label": "Index ORB 15m · intraday trend · 3R target",
             "rules": ["Nifty / Bank Nifty / Fin Nifty / Midcap Select / Sensex only (futures)",
                       "Opening range = first 15 minutes (09:15–09:30)",
                       "Long: a 5-min candle CLOSES above the range high while price is above VWAP and EMA21 > EMA50; short = mirror",
                       "Enter at the next candle's open; stop = middle of the opening range; target = 3 × risk",
                       "Skip if the stop is too tight (cost would be over 0.2R); one trade per index per side per day",
                       "No new entries after 14:30; anything open is closed at 15:15",
                       "Risk 0.5% of capital per trade; stop trading for the day after −1.5%"]}


def stats(net):
    net = np.asarray(net, float)
    if not len(net):
        return {"n": 0}
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    return {"n": int(len(net)), "win": round(float((net > 0).mean() * 100), 1), "avgR": round(float(net.mean()), 3),
            "PF": round(float(pos / neg), 2) if neg else None, "totR": round(float(net.sum()), 1)}


def per_strategy(t, label):
    t = t.copy()
    t["yr"] = pd.to_datetime(t.ts, unit="s").dt.year
    raw = t[t.trend == "DI"]
    fixed = raw[FILTERS["cost<0.2R"](raw) & (raw.mins <= 795)]
    out = {}
    for s in STRATEGIES:
        a, b = raw[raw.strat == s], fixed[fixed.strat == s]
        out[s] = {"raw": {**stats(a.rr2 - a.cost_r), "gross": round(float(a.rr2.mean()), 3) if len(a) else None,
                          "cost": round(float(a.cost_r.mean()), 3) if len(a) else None},
                  "fixed": {**stats(b.trail - b.cost_r), "gross": round(float(b.trail.mean()), 3) if len(b) else None,
                            "cost": round(float(b.cost_r.mean()), 3) if len(b) else None,
                            "years": {int(k): round(float(v), 3) for k, v in (b.trail - b.cost_r).groupby(b.yr).mean().items()}}}
    return out


def candidate_curve(t):
    c = CANDIDATE
    g = t[(t.strat == c["strat"]) & (t.trend == c["trend"])]
    g = g[FILTERS[c["filter"]](g) & g.sym.isin(TRADABLE)].sort_values("ts")
    net = g[c["exit"]] - g.cost_r
    day = pd.to_datetime(g.ts, unit="s").dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata").dt.date
    d = net.groupby(day.values).sum()
    eq = d.cumsum()
    cap, risk = 1_000_000, 0.005
    yrs = net.groupby(pd.to_datetime(g.ts, unit="s").dt.year.values).agg(["sum", "count", "mean"])
    return {"stats": stats(net), "gross": round(float(g[c["exit"]].mean()), 3), "cost": round(float(g.cost_r.mean()), 3),
            "curve": [[str(k), round(float(v), 2)] for k, v in eq.items()],
            "rupees_per_R": cap * risk, "maxDD_R": round(float((eq.cummax() - eq).max()), 1) if len(eq) else 0,
            "years": {int(k): {"R": round(float(r["sum"]), 1), "n": int(r["count"]), "avgR": round(float(r["mean"]), 3)} for k, r in yrs.iterrows()}}


def build():
    t60 = pd.read_pickle(B.ROOT.parent / "trades_60m.pkl")
    t5 = pd.read_pickle(B.ROOT.parent / "trades_5m.pkl")
    joint = pd.read_pickle(B.ROOT.parent / "joint.pkl") if (B.ROOT.parent / "joint.pkl").exists() else None
    findings = [
        ["Costs eat the edge", "A 5-minute stop is usually 0.15–0.25% of price, while one intraday round trip in India costs ~0.094% for stocks "
         "(brokerage, STT, exchange, SEBI, stamp, GST, slippage) and ~0.079% for index futures (STT on futures is now 0.05%). So costs take "
         "0.3–0.5R from every 5-minute trade. Fix used: skip any setup where costs would exceed 0.2R."],
        ["Edges before costs are tiny", "Across 2.9 million hourly trades, 74% of the strategy combinations make money BEFORE costs, but the median "
         "edge is only +0.016R per trade, about one-fifth of the cost."],
        ["Nothing survives all years", f"{len(joint) if joint is not None else '17,000+'} combinations with data on both timeframes. On 3 years of hourly bars, none "
         "had a t-stat above 3, and none was positive in every year with more than 300 trades. The best 5-minute results (t ≈ 2.4) are about what "
         "chance alone produces when you try 9,000 variants."],
        ["Switching strategies by recent form fails", "Picking each month whichever strategies did best over the last 3, 6 or 12 months lost money "
         "in every setting: strategy performance does not persist from one month to the next."],
        ["Regime filters do not rescue it", "Breakout, pullback, momentum and reversal families are negative in every India VIX band, every "
         "Nifty trendiness band and every volatility band, in every year."],
        ["Indian intraday drift is negative", "From the 09:15 open to 15:15, the average NSE stock fell 0.04–0.17% a day in every year from 2023 to 2026 "
         "(the market's gains come overnight). That is why long setups are worse than shorts here, and why buying intraday dips fails."],
        ["Late entries are worst", "Entries after 13:15 have the worst net R in every family. Fix used: no new entries after 13:15."],
        ["Gap-down reversals are real but lumpy", "Buying stocks that gap down 2%+ at the 09:15 open made money in some quarters (+1–3% a trade) and "
         "nothing in others. Entering one hour later kills it. It is regime-driven, not a steady edge."],
    ]
    out = {"built": datetime.now().isoformat(timespec="minutes"),
           "data": {"symbols_5m": int(t5.sym.nunique()), "symbols_60m": int(t60.sym.nunique()),
                    "trades_5m": int(len(t5)), "trades_60m": int(len(t60)),
                    "from_5m": str(pd.to_datetime(t5.ts.min(), unit="s").date()), "from_60m": str(pd.to_datetime(t60.ts.min(), unit="s").date()),
                    "to": str(pd.to_datetime(t5.ts.max(), unit="s").date()), "strategies": len(STRATEGIES),
                    "combos": 9080 + 8064},
           "costs": {"stock_pct": round(cost_frac("RELIANCE") * 100, 3), "index_pct": round(cost_frac("NIFTY") * 100, 3)},
           "strategies": {"h60": per_strategy(t60, "60m"), "m5": per_strategy(t5, "5m")},
           "candidate": {**CANDIDATE, "h60": candidate_curve(t60), "m5": candidate_curve(t5)},
           "findings": findings,
           "verdict": "No strategy passed: none is reliably profitable after Indian costs across 3 years. The candidate below is the closest to break-even "
                      "(about 0R over 3 years, positive in the last 60 days). It runs as a paper trade only, and its live results are logged so the "
                      "evidence keeps growing."}
    OUT.write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    o = build()
    print(json.dumps(o["candidate"]["h60"]["years"]), o["candidate"]["h60"]["stats"], o["candidate"]["m5"]["stats"])
