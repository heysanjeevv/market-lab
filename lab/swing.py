"""Swing (multi-day, delivery) backtest on 5 years of daily bars. Signal at a day's close, entry at the next
day's open, stop set before entry, exits checked day by day (a gap through the stop fills at the open),
full delivery costs. Only trades in the trend's direction (longs in an uptrend: delivery cannot short)."""
import multiprocessing as mp

import numpy as np
import pandas as pd

from . import bars as B
from .features import ema, rma, shift, supertrend

# Delivery (CNC) round trip, as a fraction: STT 0.1% on buy and on sell, stamp 0.015% on the buy, NSE txn
# 0.00297% x2, SEBI 0.0001% x2, brokerage up to Rs 20 an order (~0.02% a side on Rs 1 lakh), DP charge
# ~Rs 16 on the sell (~0.016%), GST 18% on brokerage+txn+SEBI, slippage 0.05% a side.
def cost_frac(notional=100000.0):
    brok = 2 * min(20, 0.0003 * notional) / notional * 0   # most brokers: free delivery; kept at 0, DP below
    stt, stamp, txn, sebi, dp, slip = 0.002, 0.00015, 2 * 0.0000297, 2 * 0.000001, 15.93 / notional, 0.001
    gst = 0.18 * (brok + txn + sebi + dp)
    return brok + stt + stamp + txn + sebi + dp + gst + slip


COST = cost_frac()
SKIP = set(B.INDICES)


def features(d, nifty):
    o, h, l, c, v = (d[k].values.astype(float) for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    f = {"o": o, "h": h, "l": l, "c": c, "v": v, "n": n, "date": np.array([x.date() for x in d.index])}
    tr = np.maximum(h - l, np.maximum(abs(h - shift(c, 1, c[0])), abs(l - shift(c, 1, c[0]))))
    f["atr"] = rma(tr, 14)
    s = pd.Series(c)
    for k in (5, 10, 20, 50, 100, 200):
        f[f"sma{k}"] = s.rolling(k).mean().values
    for k in (10, 20, 50):
        f[f"ema{k}"] = ema(c, k)
    dlt = np.diff(c, prepend=c[0])
    for k in (2, 14):
        f[f"rsi{k}"] = 100 - 100 / (1 + rma(np.maximum(dlt, 0), k) / np.maximum(rma(np.maximum(-dlt, 0), k), 1e-12))
    sd = s.rolling(20).std().values
    f["bbu"], f["bbl"] = f["sma20"] + 2 * sd, f["sma20"] - 2 * sd
    f["bbw"] = 4 * sd / f["sma20"]
    f["bbw_min"] = pd.Series(f["bbw"]).rolling(120).min().values
    f["hh20"] = shift(pd.Series(h).rolling(20).max().values)
    f["hh55"] = shift(pd.Series(h).rolling(55).max().values)
    f["hh252"] = shift(pd.Series(h).rolling(252).max().values)
    f["ll10"] = pd.Series(l).rolling(10).min().values
    f["ll20"] = pd.Series(l).rolling(20).min().values
    f["vavg"] = shift(pd.Series(v).rolling(50).mean().values)
    m = ema(c, 12) - ema(c, 26)
    f["macd"], f["macds"] = m, ema(m, 9)
    f["st"], f["stl"] = supertrend(h, l, c, rma(tr, 10), 3.0)
    up, dn = h - shift(h, 1, h[0]), shift(l, 1, l[0]) - l
    pdm, mdm = np.where((up > dn) & (up > 0), up, 0), np.where((dn > up) & (dn > 0), dn, 0)
    a = np.maximum(f["atr"], 1e-12)
    pdi, mdi = 100 * rma(pdm, 14) / a, 100 * rma(mdm, 14) / a
    f["adx"], f["pdi"], f["mdi"] = rma(100 * abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12), 14), pdi, mdi
    rng = h - l
    f["nr7"] = rng <= pd.Series(rng).rolling(7).min().values
    f["inside"] = (h < shift(h)) & (l > shift(l))
    f["roc63"] = s.pct_change(63).values * 100
    f["roc126"] = s.pct_change(126).values * 100
    f["gap"] = (o / shift(c) - 1) * 100
    f["chg"] = (c / shift(c) - 1) * 100
    body = abs(c - o)
    f["bull_engulf"] = (c > o) & (shift(c) < shift(o)) & (c >= shift(o)) & (o <= shift(c))
    f["hammer"] = ((np.minimum(o, c) - l) >= 2 * body) & ((h - np.maximum(o, c)) <= 0.3 * np.maximum(rng, 1e-9))
    # market regime and relative strength vs Nifty (aligned by date, yesterday's or today's close only)
    nx = nifty.reindex(f["date"])
    f["mkt_up"] = (nx["c"] > nx["sma200"]).fillna(False).values
    f["mkt_roc63"] = nx["roc63"].values
    f["rs63"] = f["roc63"] - f["mkt_roc63"]
    f["uptrend"] = (c > f["sma200"]) & (f["sma50"] > f["sma200"]) & (f["ema20"] > f["sma50"])
    return f


def nifty_table():
    d = B.load("NIFTY", "1d")
    c = d.close
    t = pd.DataFrame({"c": c.values, "sma200": c.rolling(200).mean().values, "roc63": c.pct_change(63).values * 100},
                     index=[x.date() for x in d.index])
    return t[~t.index.duplicated(keep="last")]


def X(cond):
    return np.nan_to_num(np.asarray(cond, float)) > 0


# ----------------------------------------------------------------------------- strategies: (signal, stop)
def s_breakout20(f):
    return X(f["c"] > f["hh20"]) & f["uptrend"], f["c"] - 2 * f["atr"]


def s_breakout55(f):
    return X(f["c"] > f["hh55"]) & f["uptrend"], f["c"] - 2.5 * f["atr"]


def s_52wk(f):
    return X(f["c"] > f["hh252"]) & X(f["v"] > 1.5 * f["vavg"]), f["c"] - 2.5 * f["atr"]


def s_pullback20(f):
    touch = X(f["l"] <= f["ema20"]) & X(f["c"] > f["ema20"]) & X(f["c"] > f["o"])
    return touch & f["uptrend"], np.fmin(f["l"], shift(f["l"])) - 0.5 * f["atr"]


def s_pullback50(f):
    touch = X(f["l"] <= f["sma50"] * 1.01) & X(f["c"] > f["sma50"]) & X(f["c"] > f["o"])
    return touch & X(f["c"] > f["sma200"]) & X(f["sma50"] > f["sma200"]), f["l"] - 0.5 * f["atr"]


def s_rsi2(f):
    return X(f["rsi2"] < 10) & X(f["c"] > f["sma200"]), f["c"] - 2.5 * f["atr"]


def s_3down(f):
    d3 = X(f["c"] < shift(f["c"])) & X(shift(f["c"]) < shift(f["c"], 2)) & X(shift(f["c"], 2) < shift(f["c"], 3))
    return d3 & f["uptrend"], f["c"] - 2.5 * f["atr"]


def s_bb_low(f):
    return X(f["c"] < f["bbl"]) & X(f["c"] > f["sma200"]), f["c"] - 2.5 * f["atr"]


def s_squeeze(f):
    tight = X(shift(f["bbw"]) <= shift(f["bbw_min"]) * 1.1)
    return tight & X(f["c"] > f["bbu"]) & X(f["c"] > f["sma200"]), f["sma20"] - 0.5 * f["atr"]


def s_nr7(f):
    was = shift(f["nr7"].astype(float), 1, 0) > 0
    return was & X(f["c"] > shift(f["h"])) & f["uptrend"], shift(f["l"]) - 0.3 * f["atr"]


def s_inside(f):
    was = shift(f["inside"].astype(float), 1, 0) > 0
    return was & X(f["c"] > shift(f["h"], 2)) & f["uptrend"], shift(f["l"]) - 0.3 * f["atr"]


def s_flag(f):
    pole = (shift(f["c"], 5) / shift(f["c"], 15) - 1) * 100
    fh = shift(pd.Series(f["h"]).rolling(5).max().values)
    fl = shift(pd.Series(f["l"]).rolling(5).min().values)
    ok = X(pole > 12) & X((fh - fl) < 2.0 * f["atr"]) & X(f["c"] > fh)
    return ok & X(f["c"] > f["sma50"]), fl - 0.3 * f["atr"]


def s_ema_cross(f):
    x = X(f["ema20"] > f["sma50"]) & X(shift(f["ema20"]) <= shift(f["sma50"]))
    return x & X(f["c"] > f["sma200"]), f["c"] - 2.5 * f["atr"]


def s_golden(f):
    x = X(f["sma50"] > f["sma200"]) & X(shift(f["sma50"]) <= shift(f["sma200"]))
    return x, f["c"] - 3 * f["atr"]


def s_macd(f):
    x = X(f["macd"] > f["macds"]) & X(shift(f["macd"]) <= shift(f["macds"])) & X(f["macd"] > 0)
    return x & f["uptrend"], f["c"] - 2.5 * f["atr"]


def s_supertrend(f):
    return (f["st"] > 0) & (shift(f["st"]) < 0) & X(f["c"] > f["sma200"]), f["stl"]


def s_adx(f):
    strong = X(f["adx"] > 25) & X(f["pdi"] > f["mdi"])
    dip = X(f["c"] > f["ema10"]) & X(shift(f["c"]) <= shift(f["ema10"]))
    return strong & dip & f["uptrend"], f["ll10"] - 0.3 * f["atr"]


def s_engulf(f):
    near = X(f["l"] <= f["ema20"] * 1.01)
    return f["bull_engulf"] & near & f["uptrend"], np.fmin(f["l"], shift(f["l"])) - 0.3 * f["atr"]


def s_hammer(f):
    near = X(f["l"] <= f["sma50"] * 1.01) | X(f["l"] <= f["ema20"])
    return f["hammer"] & near & f["uptrend"], f["l"] - 0.3 * f["atr"]


def s_pocket(f):
    v, c = f["v"], f["c"]
    down_v = np.where(c < shift(c), v, 0)
    mx = shift(pd.Series(down_v).rolling(10).max().values)
    return X(c > shift(c)) & X(v > mx) & X(c > f["ema10"]) & f["uptrend"] & X(f["l"] <= f["sma50"] * 1.05), f["c"] - 2 * f["atr"]


def s_gap_go(f):
    return X(f["gap"] > 3) & X(f["v"] > 2.5 * f["vavg"]) & X(f["c"] > f["o"]) & X(f["c"] > f["sma50"]), f["l"] - 0.3 * f["atr"]


def s_vcp(f):
    """Tightening near highs: 10-day range under 1.8 ATR within 8% of the 52-week high, then a break."""
    r10 = shift(pd.Series(f["h"]).rolling(10).max().values) - shift(pd.Series(f["l"]).rolling(10).min().values)
    near = X(f["c"] >= 0.92 * f["hh252"])
    hh10 = shift(pd.Series(f["h"]).rolling(10).max().values)
    return X(r10 < 1.8 * f["atr"] * 2) & near & X(f["c"] > hh10) & f["uptrend"] & X(f["v"] > 1.3 * f["vavg"]), \
        shift(pd.Series(f["l"]).rolling(10).min().values) - 0.3 * f["atr"]


def s_rs_leader(f):
    """Strong relative strength vs Nifty, bought on a close back above the 10 EMA after a dip."""
    dip = X(f["c"] > f["ema10"]) & X(shift(f["c"]) <= shift(f["ema10"]))
    return dip & X(f["rs63"] > 15) & f["uptrend"], f["ll10"] - 0.3 * f["atr"]


STRATS = {
    "20-day breakout": (s_breakout20, "trend"), "55-day breakout (Turtle)": (s_breakout55, "trend"), "52-week high + volume": (s_52wk, "trend"),
    "Pullback to 20 EMA": (s_pullback20, "trend"), "Pullback to 50 SMA": (s_pullback50, "trend"), "RSI(2) dip": (s_rsi2, "mr"),
    "3 down days in uptrend": (s_3down, "mr"), "Close under lower Bollinger": (s_bb_low, "mr"), "Volatility squeeze break": (s_squeeze, "trend"),
    "NR7 breakout": (s_nr7, "trend"), "Inside-day breakout": (s_inside, "trend"), "Bull flag": (s_flag, "trend"),
    "EMA20/SMA50 cross": (s_ema_cross, "trend"), "Golden cross": (s_golden, "trend"), "MACD cross > 0": (s_macd, "trend"),
    "Supertrend flip": (s_supertrend, "trend"), "ADX trend dip-buy": (s_adx, "trend"), "Bullish engulfing @EMA": (s_engulf, "trend"),
    "Hammer @MA": (s_hammer, "trend"), "Pocket pivot": (s_pocket, "trend"), "Earnings-style gap & go": (s_gap_go, "trend"),
    "VCP tightening break": (s_vcp, "trend"), "RS leader dip-buy": (s_rs_leader, "trend"),
}
EXITS = ["trail3", "ema20", "rr2_20", "rr3_40", "time5", "time10", "mr5"]


def follow(f, ie, stop, mode):
    """Long from day ie's open. Returns (exit day, exit price, days)."""
    o, h, l, c, atr = f["o"], f["h"], f["l"], f["c"], f["atr"]
    e = o[ie]
    r = e - stop
    n = f["n"]
    tgt = {"rr2_20": e + 2 * r, "rr3_40": e + 3 * r}.get(mode)
    maxd = {"rr2_20": 20, "rr3_40": 40, "time5": 5, "time10": 10, "mr5": 10, "trail3": 120, "ema20": 120}[mode]
    st, peak = stop, e
    for j in range(ie, min(n, ie + maxd)):
        if j > ie and o[j] <= st:
            return j, o[j]                      # gapped through the stop
        if l[j] <= st:
            return j, st
        if tgt is not None and h[j] >= tgt:
            return j, max(tgt, o[j]) if j > ie else tgt
        if mode == "mr5" and c[j] > f["sma5"][j]:
            return j, c[j]
        if mode == "ema20" and c[j] < f["ema20"][j] and j > ie:
            return j, c[j]
        if mode == "trail3":
            peak = max(peak, c[j])
            if peak >= e + r:
                st = max(st, e)
            st = max(st, peak - 3 * atr[j])
    j = min(n - 1, ie + maxd - 1)
    return j, c[j]


def run_symbol(sym, nifty, start=260):
    d = B.load(sym, "1d")
    if d is None or len(d) < 400:
        return []
    d = d[d.volume > 0]
    d = d[~d.index.duplicated(keep="last")]
    f = features(d, nifty)
    bad = np.abs(np.nan_to_num(f["chg"])) > 35          # splits / bad prints
    rows = []
    for name, (fn, kind) in STRATS.items():
        sig, stop = fn(f)
        sig = np.asarray(sig, bool)
        sig[:start] = False
        for mode in EXITS:
            if kind == "trend" and mode == "mr5":
                continue
            busy = -1
            for i in np.nonzero(sig)[0]:
                ie = i + 1
                if ie >= f["n"] - 1 or ie <= busy or not np.isfinite(stop[i]):
                    continue
                e = f["o"][ie]
                r = e - stop[i]
                if r <= 0 or r / e > 0.25 or bad[max(0, i - 5):ie + 1].any():
                    continue
                j, px = follow(f, ie, stop[i], mode)
                if bad[ie:j + 1].any():
                    continue
                busy = j
                g = (px - e) / e
                rows.append((sym, name, mode, f["date"][ie], f["date"][j], j - ie + 1, g * 100, (g - COST) * 100, (g - COST) * e / r,
                             r / e * 100, bool(f["mkt_up"][i]), float(f["rs63"][i]) if np.isfinite(f["rs63"][i]) else np.nan,
                             float(f["roc126"][i]) if np.isfinite(f["roc126"][i]) else np.nan, float(f["c"][i] * f["vavg"][i] / 1e7) if np.isfinite(f["vavg"][i]) else np.nan))
    return rows


COLS = ["sym", "strat", "exit", "d_in", "d_out", "days", "gross", "net", "R", "risk", "mkt_up", "rs63", "roc126", "val_cr"]
_N = {}


def _work(sym):
    try:
        return run_symbol(sym, _N["n"])
    except Exception as e:
        print("err", sym, e, flush=True)
        return []


def run_all(procs=7):
    _N["n"] = nifty_table()
    syms = [p.stem for p in sorted((B.ROOT / "1d").glob("*.pkl")) if p.stem not in SKIP]
    out = []
    if "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(procs) as pool:
            for rows in pool.imap_unordered(_work, syms, chunksize=4):
                out.extend(rows)
    else:                                  # Windows: one process (slower, same result)
        for s in syms:
            out.extend(_work(s))
    t = pd.DataFrame(out, columns=COLS)
    t.to_pickle(B.ROOT.parent / "swing_trades.pkl")
    return t


if __name__ == "__main__":
    import time
    t0 = time.time()
    t = run_all()
    print(len(t), "trades", t.sym.nunique(), "stocks", round(time.time() - t0), "s")
