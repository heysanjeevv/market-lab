"""Backtest engine. Signals come from lab.strategies (longs on the chart, shorts on its mirror), only in the
trend's direction, entered at the NEXT bar's open, with the stop placed before entry. Every trade is then
followed bar by bar to its exit under several exit rules at once, and charged the full Indian round-trip
cost (brokerage, STT, exchange, SEBI, stamp duty, GST, plus slippage). Results are in R (multiples of the
risk taken), net of costs."""
import multiprocessing as mp

import numpy as np
import pandas as pd

from . import bars as B
from . import features as F
from .strategies import STRATEGIES

INDEX_SYMS = set(B.INDICES) - {"INDIAVIX"}
MODES = ["rr1", "rr15", "rr2", "rr3", "be2", "trail", "scale", "eod"]


def cost_frac(sym, notional=250000.0):
    """Round-trip cost as a fraction of price for one intraday trade.
    Stocks (MIS cash): brokerage min(Rs 20, 0.03%) per order, STT 0.025% on the sell, NSE txn 0.00297% x2,
    SEBI 0.0001% x2, stamp 0.003% on the buy, GST 18% on brokerage+txn+SEBI, slippage 0.02% per side.
    Index (futures): brokerage Rs 20 per order on ~Rs 15 lakh, STT 0.05% on the sell, txn 0.00173% x2,
    stamp 0.002% on the buy, GST, slippage 0.01% per side."""
    if sym in INDEX_SYMS:
        brok = 40 / 1.5e6
        txn, stt, stamp, slip = 2 * 0.0000173, 0.0005, 0.00002, 0.0002
    else:
        brok = 2 * min(20, 0.0003 * notional) / notional
        txn, stt, stamp, slip = 2 * 0.0000297, 0.00025, 0.00003, 0.0004
    sebi = 2 * 0.000001
    gst = 0.18 * (brok + txn + sebi)
    return brok + txn + stt + stamp + sebi + gst + slip


def trend_mask(f, kind):
    intra = (f["c"] > f["vwap"]) & (f["ema21"] > f["ema50"])
    if kind == "D":
        return f["dup"]
    if kind == "I":
        return intra
    if kind == "DI":
        return f["dup"] & intra
    if kind == "DIX":
        return f["dup"] & intra & f["ix_up"]
    raise ValueError(kind)


def windows(f):
    tf = f["tf"]
    entry_ok = (f["mins"] >= 555 + tf) & (f["mins"] <= 870)            # 09:15+1 bar .. 14:30
    # last bar whose close is at or before 15:15 (square off before the broker's MIS cut-off)
    exit_idx = pd.Series(np.where(f["mins"] + tf <= 915, np.arange(f["n"]), -1)).groupby(f["day"]).max().values
    return entry_ok, exit_idx


def follow(f, i_e, s, e, r, xe):
    """Outcome in gross R of one long from bar i_e (entry at its open) for every exit rule."""
    H, L, C = f["h"][i_e:xe + 1], f["l"][i_e:xe + 1], f["c"][i_e:xe + 1]
    A = f["atr"][i_e:xe + 1]
    n = len(H)
    hit = L <= s
    ks = int(np.argmax(hit)) if hit.any() else n
    eod = (C[-1] - e) / r
    out = {}
    for name, rr in (("rr1", 1), ("rr15", 1.5), ("rr2", 2), ("rr3", 3)):
        th = H >= e + rr * r
        kt = int(np.argmax(th)) if th.any() else n
        out[name] = -1.0 if ks <= kt and ks < n else rr if kt < n else eod
    out["eod"] = -1.0 if ks < n else eod
    # 1R reached -> stop to break-even; then 2R target (be2) or 3R on the second half (scale)
    t1 = H >= e + r
    k1 = int(np.argmax(t1)) if t1.any() else n
    if ks <= k1 and ks < n:
        out["be2"] = -1.0
        out["scale"] = -1.0
    elif k1 >= n:
        out["be2"] = out["scale"] = eod
    else:
        def after(target):
            for j in range(k1, n):
                if j > k1 and L[j] <= e:
                    return 0.0
                if H[j] >= e + target * r:
                    return float(target)
            return max(eod, 0.0) if L[k1:].min() > e else 0.0
        out["be2"] = after(2)
        out["scale"] = 0.5 + 0.5 * after(3)
    # chandelier trail: 2.5 ATR under the highest high, break-even after 1R
    stop, peak = s, e
    res = eod
    for j in range(n):
        if L[j] <= stop:
            res = (min(stop, f["o"][i_e + j]) - e) / r if j > 0 else (stop - e) / r
            break
        peak = max(peak, H[j])
        if peak >= e + r:
            stop = max(stop, e)
        stop = max(stop, peak - 2.5 * A[j])
    out["trail"] = res
    mx = H[:ks + 1].max() if ks < n else H.max()
    out["mfe"] = (mx - e) / r
    return out


CTX = ["rvol", "chg", "rs", "gap", "datr_pct", "adx", "d_ret5"]


def run_symbol(sym, f, fm, cost, trends=("D", "I", "DI"), strategies=None, max_per_day=1):
    rows = []
    entry_ok, exit_idx = windows(f)
    for name, fn in (strategies or STRATEGIES).items():
        for side, g in ((1, f), (-1, fm)):
            try:
                sig, stop = fn(g)
            except Exception as ex:      # a broken strategy must not kill the run
                print("strategy error", name, sym, ex)
                continue
            sig = np.asarray(sig, bool) & entry_ok
            for tk in trends:
                idx = np.nonzero(sig & trend_mask(g, tk))[0]
                taken = {}
                for i in idx:
                    ie = i + 1
                    if ie >= g["n"] or g["day"][ie] != g["day"][i] or taken.get(g["day"][i], 0) >= max_per_day:
                        continue
                    s, e = stop[i], g["o"][ie]
                    if not np.isfinite(s) or e - s <= 0:
                        continue
                    xe = exit_idx[g["day"][i]]
                    if xe < ie:
                        continue
                    r = e - s
                    taken[g["day"][i]] = taken.get(g["day"][i], 0) + 1
                    o = follow(g, ie, s, e, r, xe)
                    row = {"sym": sym, "strat": name, "trend": tk, "side": side, "ts": int(g["ts"][ie]),
                           "entry": abs(e), "risk_pct": r / abs(e) * 100, "cost_r": cost * abs(e) / r,
                           "mins": int(g["mins"][ie]), "ix_al": bool(g["ix_up"][i]), "atr_pct": g["atr"][i] / abs(g["c"][i]) * 100}
                    for m in MODES:
                        row[m] = o[m]
                    row["mfe"] = o["mfe"]
                    for cx in CTX:
                        row[cx] = float(g[cx][i])
                    rows.append(row)
    return rows


_G = {}


def prepare(sym, tf="5m"):
    df = B.load(sym, tf)
    daily = B.load(sym, "1d")
    if df is None or len(df) < 300:
        return None
    tfm = 5 if tf == "5m" else 60 if tf == "60m" else 15
    ix = _G.get(("ix", tf))
    f = F.build(df, daily, tfm, ix[0] if ix else None, 1)
    fm = F.build(F.mirror(df), F.mirror(daily) if daily is not None else None, tfm, ix[1] if ix else None, -1)
    return f, fm


def set_index(tf="5m"):
    _G.pop(("ix", tf), None)
    p = prepare("NIFTY", tf)
    _G[("ix", tf)] = (F.index_context(p[0]), F.index_context(p[1]))


def _work(args):
    sym, tf, trends = args
    try:
        p = prepare(sym, tf)
        if p is None:
            return []
        return run_symbol(sym, p[0], p[1], cost_frac(sym), trends)
    except Exception as ex:
        print("symbol error", sym, ex, flush=True)
        return []


def run_all(syms, tf="5m", trends=("D", "I", "DI"), procs=7):
    syms = [s for s in syms if s != "INDIAVIX"]
    set_index(tf)
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    out = []
    with ctx.Pool(procs) as pool:
        for i, rows in enumerate(pool.imap_unordered(_work, [(s, tf, trends) for s in syms], chunksize=2)):
            out.extend(rows)
            if i % 50 == 0:
                print(f"{i}/{len(syms)} symbols, {len(out)} trades", flush=True)
    t = pd.DataFrame(out)
    for c in t.columns:
        if t[c].dtype == "float64":
            t[c] = t[c].astype("float32")
    return t
