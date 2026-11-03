"""Strategy trades for any chart (any symbol, any intraday timeframe, any of the 39 strategies), drawn with
entry, stop, target and result; plus the paper-trade journal of the candidate system on the indices,
and a daily job that keeps extending the 5-minute history beyond Yahoo's 60-day window."""
import json
import threading
import time
import urllib.parse

import numpy as np
import pandas as pd

from sm import market as M
from . import bars as B
from . import features as F
from .engine import cost_frac, trend_mask, windows
from .report import CANDIDATE
from .strategies import STRATEGIES

RANGE = {"1m": "5d", "5m": "60d", "15m": "60d", "60m": "730d"}
TFMIN = {"1m": 1, "5m": 5, "15m": 15, "60m": 60}
EXITS = {"rr15": 1.5, "rr2": 2.0, "rr3": 3.0, "trail": None, "eod": None}
_CACHE, _LOCK = {}, threading.Lock()
JOURNAL = M.DATA / "paper_journal.json"
PAPER_SYMS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]


def bars(sym, tf):
    """Cached history merged with a fresh Yahoo pull (30 s memo)."""
    key = (sym, tf)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < 30:
            return hit[1]
    y = urllib.parse.quote(B.yahoo(sym))
    d = M.get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{y}?interval={tf}&range={RANGE[tf]}", 20)["chart"]["result"][0]
    o = d["indicators"]["quote"][0]
    df = pd.DataFrame({k: o[k] for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(d["timestamp"], unit="s", utc=True).tz_convert(M.IST)).dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)
    t = df.index.hour * 60 + df.index.minute
    df = df[(t >= 555) & (t < 930)]
    old = B.load(sym, tf) if tf in ("5m", "60m") else None
    if old is not None:
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    with _LOCK:
        _CACHE[key] = (time.time(), df)
    return df


def daily(sym):
    p = B.path(sym, "1d")
    if not p.exists() or time.time() - p.stat().st_mtime > 18 * 3600:
        try:
            B.save(sym, "1d", B.download(sym, "1d"))
        except Exception:
            pass
    return B.load(sym, "1d")


def _features(sym, tf):
    df = bars(sym, tf)
    if df is None or len(df) < 60:
        return None
    tfm = TFMIN[tf]
    ix = None
    if sym != "NIFTY" and sym not in B.INDICES:
        try:
            n = bars("NIFTY", tf)
            nf = F.build(n, daily("NIFTY"), tfm, None, 1)
            nfm = F.build(F.mirror(n), F.mirror(daily("NIFTY")), tfm, None, -1)
            ix = (F.index_context(nf), F.index_context(nfm))
        except Exception:
            ix = None
    dly = daily(sym)
    f = F.build(df, dly, tfm, ix[0] if ix else None, 1)
    fm = F.build(F.mirror(df), F.mirror(dly) if dly is not None else None, tfm, ix[1] if ix else None, -1)
    return f, fm


def _walk(g, ie, s, e, r, xe, mode):
    """Bar-by-bar exit of one long (on g's orientation): returns (exit bar, exit price, reason, target)."""
    H, L, C, O, A = g["h"], g["l"], g["c"], g["o"], g["atr"]
    rr = EXITS[mode]
    tgt = e + rr * r if rr else None
    stop, peak = s, e
    for j in range(ie, xe + 1):
        if L[j] <= stop:
            px = min(stop, O[j]) if j > ie else stop
            return j, px, "trail stop" if stop > s else "stop", tgt
        if tgt is not None and H[j] >= tgt:
            return j, tgt, "target", tgt
        if mode == "trail":
            peak = max(peak, H[j])
            if peak >= e + r:
                stop = max(stop, e)
            stop = max(stop, peak - 2.5 * A[j])
    return xe, C[xe], "15:15 exit", tgt


def chart_trades(sym, tf="5m", strat=None, trend=None, exit_mode=None, max_cost_r=0.2):
    if tf not in TFMIN:
        return {"trades": [], "note": "Trades are intraday: open the 1m, 5m, 15m or 1H chart."}
    cand = strat in (None, "", "candidate")
    strat = CANDIDATE["strat"] if cand else strat
    trend = trend or (CANDIDATE["trend"] if cand else "DI")
    exit_mode = exit_mode or (CANDIDATE["exit"] if cand else "rr2")
    fn = STRATEGIES.get(strat)
    if fn is None:
        return {"error": f"unknown strategy {strat}"}
    p = _features(sym, tf)
    if p is None:
        return {"trades": [], "note": "Not enough bars."}
    cost = cost_frac(sym)
    out = []
    for side, g in ((1, p[0]), (-1, p[1])):
        entry_ok, exit_idx = windows(g)
        sig, stop = fn(g)
        sig = np.asarray(sig, bool) & entry_ok & trend_mask(g, trend)
        taken = set()
        for i in np.nonzero(sig)[0]:
            ie = i + 1
            d = g["day"][i]
            if d in taken:
                continue
            last_bar = ie >= g["n"]
            if not last_bar and g["day"][ie] != d:
                continue
            s = stop[i]
            if not np.isfinite(s):
                continue
            if last_bar:          # signal on the newest closed bar: the trade starts at the next open
                e = g["c"][i]
                r = e - s
                if r <= 0 or cost * abs(e) / r > max_cost_r:
                    continue
                tgt = e + EXITS[exit_mode] * r if EXITS[exit_mode] else None
                out.append({"side": side, "t_in": int(g["ts"][i]) + TFMIN[tf] * 60, "p_in": round(side * e, 2), "stop": round(side * s, 2),
                            "target": round(side * tgt, 2) if tgt else None, "status": "SIGNAL", "reason": "enter at next open"})
                continue
            e = g["o"][ie]
            r = e - s
            if r <= 0 or cost * abs(e) / r > max_cost_r:
                continue
            xe = exit_idx[d]
            if xe < ie:
                continue
            taken.add(d)
            today_open = xe >= g["n"] - 1 and g["mins"][g["n"] - 1] + TFMIN[tf] < 915
            j, px, why, tgt = _walk(g, ie, s, e, r, min(xe, g["n"] - 1), exit_mode)
            gross = (px - e) / r
            net = gross - cost * abs(e) / r
            live = today_open and why == "15:15 exit"
            out.append({"side": side, "t_in": int(g["ts"][ie]), "p_in": round(side * e, 2), "stop": round(side * s, 2),
                        "target": round(side * tgt, 2) if tgt else None, "t_out": int(g["ts"][j]), "p_out": round(side * px, 2),
                        "R": round(float(net), 2), "gross": round(float(gross), 2), "reason": "open" if live else why,
                        "status": "OPEN" if live else "CLOSED", "risk_pct": round(r / abs(e) * 100, 2)})
    out.sort(key=lambda x: x["t_in"])
    closed = [x["R"] for x in out if x["status"] == "CLOSED"]
    summ = {"n": len(closed), "win": round(100 * np.mean([r > 0 for r in closed]), 1) if closed else None,
            "avgR": round(float(np.mean(closed)), 3) if closed else None, "totR": round(float(np.sum(closed)), 1) if closed else 0}
    return {"sym": sym, "tf": tf, "strat": strat, "trend": trend, "exit": exit_mode, "candidate": cand,
            "cost_pct": round(cost * 100, 3), "trades": out, "summary": summ}


# ----------------------------------------------------------------------------- paper journal
def journal():
    try:
        return json.loads(JOURNAL.read_text())
    except Exception:
        return {"trades": []}


def paper_tick():
    j = journal()
    seen = {(x["sym"], x["t_in"], x["side"]): x for x in j["trades"]}
    for sym in PAPER_SYMS:
        try:
            r = chart_trades(sym, "5m")
        except Exception:
            continue
        today = pd.Timestamp.now(tz=M.IST).normalize().timestamp()
        for t in r.get("trades", []):
            if t["t_in"] < today - 86400 * 3 or t["status"] == "SIGNAL":
                continue
            k = (sym, t["t_in"], t["side"])
            seen[k] = {**seen.get(k, {}), **t, "sym": sym, "logged": seen.get(k, {}).get("logged") or int(time.time())}
    j["trades"] = sorted(seen.values(), key=lambda x: x["t_in"])
    j["updated"] = int(time.time())
    JOURNAL.write_text(json.dumps(j))
    return j


def run(log=print):
    last_extend = None
    while True:
        try:
            now = M.now_ist()
            mins = now.hour * 60 + now.minute
            if now.weekday() < 5 and 555 <= mins <= 935:
                paper_tick()
            if now.weekday() < 5 and mins >= 960 and last_extend != now.date():
                last_extend = now.date()
                syms = [s for s in sorted(p.stem for p in (B.ROOT / "5m").glob("*.pkl"))]
                log(f"extending 5-minute history for {len(syms)} symbols")
                B.fetch_all(syms, tfs=("5m", "60m", "1d"), fresh_hours=10, log=lambda m: None)
                log("history extended")
        except Exception as e:
            log(f"lab loop error: {e}")
        time.sleep(60)
