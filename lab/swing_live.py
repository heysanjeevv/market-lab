"""Swing system live: trades for the daily chart, tomorrow's buy list and the model portfolio's open
positions, refreshed after each close; plus the research summary for the Lab panel."""
import json
import time

import numpy as np
import pandas as pd

from sm import market as M
from . import bars as B
from . import swing as S
from . import swing_pf as P

CAND = {"strat": "VCP tightening break", "exit": "rr3_40", "k": 10,
        "label": "VCP tightening break · Nifty above 200-DMA · 3R / 40 days",
        "rules": ["Universe: Nifty 500 + F&O stocks, delivery (CNC), longs only",
                  "Market filter: buy only while Nifty closes above its 200-day average",
                  "Stock trend: close > 200 SMA, 50 SMA > 200 SMA, 20 EMA > 50 SMA",
                  "Setup: within 8% of the 52-week high, the last 10 days' range under 3.6 × ATR (tightening)",
                  "Trigger: close above the 10-day high on volume ≥ 1.3 × the 50-day average; buy next day's open",
                  "Stop: under the 10-day low (−0.3 ATR); target 3 × risk; exit after 40 trading days at the latest",
                  "Up to 10 positions of 10% each; when several trigger, take the strongest vs Nifty (63-day relative strength) first"]}
OUT = M.DATA / "swing_summary.json"
STATE = {"live": None, "t": 0}
EXIT_NAMES = {"trail3": "3-ATR trailing stop", "ema20": "close under 20 EMA", "rr2_20": "2R or 20 days", "rr3_40": "3R or 40 days",
              "time5": "5 days", "time10": "10 days", "mr5": "close above 5 SMA"}


def _load(sym):
    p = B.path(sym, "1d")
    if not p.exists() or time.time() - p.stat().st_mtime > 18 * 3600:
        try:
            B.save(sym, "1d", B.download(sym, "1d"))
        except Exception:
            pass
    d = B.load(sym, "1d")
    if d is None:
        return None
    d = d[d.volume > 0]
    return d[~d.index.duplicated(keep="last")]


def setup_state(f):
    """Where the latest close stands in the VCP setup: forming (tight near highs, waiting for the trigger),
    triggered (fired at this close), or nothing."""
    i = f["n"] - 1
    h, l, c, atr = f["h"], f["l"], f["c"], f["atr"]
    hh10, ll10 = float(h[i - 9:i + 1].max()), float(l[i - 9:i + 1].min())
    tight = (hh10 - ll10) < 3.6 * atr[i]
    near = c[i] >= 0.92 * f["hh252"][i] if np.isfinite(f["hh252"][i]) else False
    trend = bool(f["uptrend"][i])
    base = {"close": round(float(c[i]), 2), "atr_pct": round(float(atr[i] / c[i] * 100), 2), "rs63": round(float(f["rs63"][i]), 1) if np.isfinite(f["rs63"][i]) else None,
            "from_high": round(float((c[i] / f["hh252"][i] - 1) * 100), 1) if np.isfinite(f["hh252"][i]) else None, "trend": trend, "tight": bool(tight), "near_high": bool(near),
            "mkt_up": bool(f["mkt_up"][i])}
    if trend and near and tight:
        trig = hh10 + 0.05
        stop = ll10 - 0.3 * atr[i]
        r = trig - stop
        return {**base, "state": "FORMING", "trigger": round(trig, 2), "stop": round(float(stop), 2), "target": round(float(trig + 3 * r), 2),
                "risk_pct": round(float(r / trig * 100), 2), "dist_pct": round(float((trig / c[i] - 1) * 100), 2),
                "vol_need": round(float(1.3 * f["vavg"][i]), 0) if np.isfinite(f["vavg"][i]) else None}
    return {**base, "state": "NONE"}


def trades_for(sym, strat=None, exit_mode=None, nifty=None):
    strat = strat if strat in S.STRATS else CAND["strat"]
    exit_mode = exit_mode if exit_mode in S.EXITS else CAND["exit"] if strat == CAND["strat"] else "trail3"
    d = _load(sym)
    if d is None or len(d) < 260:
        return {"trades": [], "note": "Not enough daily history for a swing read."}
    f = S.features(d, nifty if nifty is not None else S.nifty_table())
    sig, stop = S.STRATS[strat][0](f)
    sig = np.asarray(sig, bool)
    sig[:220] = False
    ts = np.asarray(d.index.tz_convert("UTC").tz_localize(None).values.astype("datetime64[s]").astype("int64"))
    out, busy, n = [], -1, f["n"]
    for i in np.nonzero(sig)[0]:
        if i <= busy or not np.isfinite(stop[i]):
            continue
        if i == n - 1:      # fired on the latest close: buy at the next open
            e, r = f["c"][i], f["c"][i] - stop[i]
            if r > 0:
                out.append({"side": 1, "t_in": int(ts[i]) + 86400, "p_in": round(float(e), 2), "stop": round(float(stop[i]), 2),
                            "target": round(float(e + 3 * r), 2) if exit_mode == "rr3_40" else round(float(e + 2 * r), 2) if exit_mode == "rr2_20" else None,
                            "status": "SIGNAL", "mkt_up": bool(f["mkt_up"][i]), "rs63": round(float(f["rs63"][i]), 1)})
            continue
        ie = i + 1
        e = f["o"][ie]
        r = e - stop[i]
        if r <= 0 or r / e > 0.25:
            continue
        j, px = S.follow(f, ie, stop[i], exit_mode)
        busy = j
        maxd = {"rr2_20": 20, "rr3_40": 40, "time5": 5, "time10": 10, "mr5": 10, "trail3": 120, "ema20": 120}[exit_mode]
        live = j == n - 1 and j - ie + 1 < maxd and abs(px - f["c"][j]) < 1e-9
        tgt = {"rr2_20": e + 2 * r, "rr3_40": e + 3 * r}.get(exit_mode)
        g = (px - e) / e
        net = g - S.COST
        out.append({"side": 1, "t_in": int(ts[ie]), "p_in": round(float(e), 2), "stop": round(float(stop[i]), 2),
                    "target": round(float(tgt), 2) if tgt else None, "t_out": int(ts[j]), "p_out": round(float(px), 2),
                    "R": round(float(net * e / r), 2), "pct": round(float(net * 100), 2), "days": int(j - ie + 1),
                    "status": "OPEN" if live else "CLOSED", "mkt_up": bool(f["mkt_up"][i]), "rs63": round(float(f["rs63"][i]), 1)})
    closed = [x for x in out if x["status"] == "CLOSED"]
    rs = [x["R"] for x in closed]
    state = setup_state(f) if strat == CAND["strat"] else None
    if state is not None and out and out[-1]["status"] in ("SIGNAL", "OPEN"):
        state = {**state, "state": out[-1]["status"], **{k: out[-1][k] for k in ("p_in", "stop", "target")}}
    return {"sym": sym, "tf": "1d", "state": state, "days_held": [x["days"] for x in closed], "strat": strat, "exit": exit_mode, "exit_name": EXIT_NAMES[exit_mode], "swing": True,
            "candidate": strat == CAND["strat"] and exit_mode == CAND["exit"], "cost_pct": round(S.COST * 100, 3), "trades": out,
            "summary": {"n": len(rs), "win": round(100 * np.mean([x > 0 for x in rs]), 1) if rs else None,
                        "avgR": round(float(np.mean(rs)), 3) if rs else None, "totR": round(float(np.sum(rs)), 1) if rs else 0,
                        "pct": round(float(np.sum([x["pct"] for x in closed])), 1) if closed else 0}}


def scan(log=print):
    """Tomorrow's buy list and the open positions of the candidate, over the whole universe."""
    nifty = S.nifty_table()
    sig, opn, forming, recent = [], [], [], []
    cut = time.time() - 15 * 86400
    syms = [p.stem for p in sorted((B.ROOT / "1d").glob("*.pkl")) if p.stem not in S.SKIP]
    for s in syms:
        try:
            r = trades_for(s, CAND["strat"], CAND["exit"], nifty)
        except Exception:
            continue
        for t in r["trades"]:
            if t["status"] == "SIGNAL":
                sig.append({**t, "sym": s})
            elif t["status"] == "OPEN" and t["mkt_up"]:
                opn.append({**t, "sym": s})
            elif t["status"] == "CLOSED" and t["t_in"] >= cut:
                recent.append({**t, "sym": s})
        st = r.get("state") or {}
        if st.get("state") == "FORMING":
            forming.append({**st, "sym": s})
    n = nifty.dropna()
    up = bool(n.c.iloc[-1] > n.sma200.iloc[-1])
    STATE["live"] = {"time": int(time.time()), "nifty": round(float(n.c.iloc[-1]), 1), "nifty_200": round(float(n.sma200.iloc[-1]), 1),
                     "mkt_up": up, "signals": sorted(sig, key=lambda x: -x["rs63"]), "open": sorted(opn, key=lambda x: -x["t_in"]),
                     "forming": sorted(forming, key=lambda x: x["dist_pct"]), "recent": sorted(recent, key=lambda x: -x["t_in"])}
    log(f"swing scan: {len(forming)} forming, {len(sig)} signals, {len(opn)} open, market {'up' if up else 'down'}")
    return STATE["live"]


def build_summary():
    """Research numbers for the Lab (run after python -m lab.swing)."""
    from . import swing_an as A
    t = pd.read_pickle(B.ROOT.parent / "swing_trades.pkl")
    ew_i, _ = A.ew_index()
    T, tt = A.table(t, ew_i)
    rows = {}
    for (s, x), r in T.iterrows():
        cur = rows.get(s)
        if cur is None or r["excess%"] > cur["excess"]:
            rows[s] = {"exit": x, "exit_name": EXIT_NAMES[x], "n": int(r.n), "win": float(r["win%"]), "net": float(r["net%"]), "excess": float(r["excess%"]),
                       "days": float(r.days), "IS": float(r.IS_net), "OOS": float(r.OOS_net), "OOS_exc": float(r.OOS_exc),
                       "years": {int(y): float(r[y]) for y in (2022, 2023, 2024, 2025, 2026) if y in r and np.isfinite(r[y])}}
    working = {k: v for k, v in rows.items() if v["excess"] >= 0.25 and v["OOS_exc"] > 0 and v["net"] > 0}
    removed = sorted(set(rows) - set(working))
    profile, best = stock_profile(t)
    O, C = P.panels()
    start = "2022-11-01"
    m = (t.strat == CAND["strat"]) & (t.exit == CAND["exit"]) & t.mkt_up
    e, log = P.simulate(t[m], O, C, k=CAND["k"], start=start)
    nb, ew = P.bench(C, start)
    nt = S.nifty_table()
    regime = [[str(k), bool(nt.c.get(k, np.nan) > nt.sma200.get(k, np.nan))] for k in e.dropna().index]
    skipped = t[(t.strat == CAND["strat"]) & (t.exit == CAND["exit"]) & ~t.mkt_up & (pd.to_datetime(t.d_in) >= pd.Timestamp(start))]
    def curve(x):
        x = x.dropna()
        return [[str(k), round(float(v / x.iloc[0] * 100), 2)] for k, v in x.items()]
    out = {"built": pd.Timestamp.now().isoformat(timespec="minutes"), "cost_pct": round(S.COST * 100, 3), "stocks": int(t.sym.nunique()),
           "trades": int(len(t)), "strategies": working, "removed": removed, "profile": profile, "best_stocks": best, "candidate": {**CAND, "metrics": P.metrics(e.copy()), "nifty": P.metrics(nb.copy()),
           "ew": P.metrics(ew.copy()), "curve": curve(e), "regime": regime, "skipped": {"n": int(len(skipped)), "avg": round(float(skipped.net.mean()), 2) if len(skipped) else None},
           "curve_nifty": curve(nb), "curve_ew": curve(ew), "trades": int(len(log)),
           "win": round(float((log.pnl > 0).mean() * 100), 1) if len(log) else None},
           "robust": []}
    OUT.write_text(json.dumps(out, default=float))
    return out


def stock_profile(t):
    """Which kinds of stocks the system's own trades (VCP, 3R/40d, market filter on) worked best in."""
    v = t[(t.strat == CAND["strat"]) & (t.exit == CAND["exit"]) & t.mkt_up].copy()
    try:
        cls = json.loads((M.DATA / "classification.json").read_text())
    except Exception:
        cls = {}
    v["sector"] = v.sym.map(lambda x: (cls.get(x) or {}).get("sector") or "Other")
    mc = v.sym.map(lambda x: ((cls.get(x) or {}).get("mcap") or 0) / 1e7)
    v["size"] = np.where(mc >= 50000, "Large cap (₹50k cr+)", np.where(mc >= 15000, "Mid cap (₹15–50k cr)", np.where(mc > 0, "Small cap (< ₹15k cr)", "Unknown")))
    v["liq"] = pd.cut(v.val_cr, [0, 10, 30, 100, 1e9], labels=["Turnover < ₹10 cr/day", "₹10–30 cr/day", "₹30–100 cr/day", "₹100 cr+/day"])
    v["vol"] = pd.cut(v.risk, [0, 6, 9, 13, 100], labels=["Tight base (risk < 6%)", "Risk 6–9%", "Risk 9–13%", "Wide base (risk > 13%)"])
    v["rsb"] = pd.cut(v.rs63, [-999, 0, 15, 30, 999], labels=["Weaker than Nifty", "RS 0–15", "RS 15–30", "RS 30+ (leaders)"])
    out = {}
    for col, title in (("sector", "Sector"), ("size", "Company size"), ("liq", "Liquidity"), ("vol", "Base width"), ("rsb", "Strength vs Nifty (3 months)")):
        g = v.groupby(col, observed=True).agg(n=("net", "size"), avg=("net", "mean"), win=("net", lambda x: (x > 0).mean() * 100))
        g = g[g.n >= (25 if col == "sector" else 40)].sort_values("avg", ascending=False)
        out[title] = [{"k": str(k), "n": int(r.n), "avg": round(float(r.avg), 2), "win": round(float(r.win), 1)} for k, r in g.iterrows()]
    s = v.groupby("sym").agg(n=("net", "size"), avg=("net", "mean"), tot=("net", "sum"), win=("net", lambda x: (x > 0).mean() * 100))
    s = s[s.n >= 4].sort_values(["avg"], ascending=False).head(40)
    best = [{"sym": k, "n": int(r.n), "avg": round(float(r.avg), 2), "tot": round(float(r.tot), 1), "win": round(float(r.win), 1),
             "sector": (cls.get(k) or {}).get("sector", "")} for k, r in s.iterrows()]
    return out, best


def summary():
    try:
        s = json.loads(OUT.read_text())
    except Exception:
        s = {"error": "run: python -m lab.setup"}
    # on a fresh install the backtest runs before the company classification exists: redo the stock profile once it does
    prof = s.get("profile") or {}
    if prof and all(r.get("k") in ("Other", "Unknown") for r in (prof.get("Sector", []) + prof.get("Company size", []))) \
            and (M.DATA / "classification.json").exists():
        try:
            s["profile"], s["best_stocks"] = stock_profile(pd.read_pickle(B.ROOT.parent / "swing_trades.pkl"))
            OUT.write_text(json.dumps({k: v for k, v in s.items() if k not in ("live", "list", "exits")}, default=float))
        except Exception:
            pass
    s["live"] = STATE["live"]
    s["list"] = list(S.STRATS)
    s["exits"] = EXIT_NAMES
    return s


def run(log=print):
    last = None
    while True:
        try:
            now = M.now_ist()
            key = (now.date(), now.hour >= 16)
            if STATE["live"] is None or (key != last and now.hour >= 16):
                last = key
                scan(log)
        except Exception as e:
            log(f"swing scan error: {e}")
        time.sleep(300)
