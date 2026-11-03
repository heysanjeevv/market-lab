"""The strategy library. Each one is written ONCE, for longs: fn(f) -> (signal, stop) arrays, where a signal at
bar i is acted on at bar i+1's open. Shorts are the same function run on the mirrored chart (see
features.mirror). The trend filter, the entry window and the exits are applied by the engine, not here."""
import numpy as np
import pandas as pd

from .features import shift


def cross_up(a, b):
    a, b = np.asarray(a, float), np.broadcast_to(np.asarray(b, float), np.shape(a))
    return (a > b) & (shift(a) <= shift(b))


def nz(x):
    return np.nan_to_num(np.asarray(x, float), nan=0.0) > 0


def mn(*xs):
    return np.fmin.reduce(xs)


# ----------------------------------------------------------------------------- breakouts
def orb15(f):
    return nz(cross_up(f["c"], f["or15h"])), (f["or15h"] + f["or15l"]) / 2


def orb30(f):
    return nz(cross_up(f["c"], f["or30h"])), (f["or30h"] + f["or30l"]) / 2


def pdh_break(f):
    return nz(cross_up(f["c"], f["pdh"])) & (f["k"] > 0), f["low3"] - 0.2 * f["atr"]


def cpr_break(f):
    narrow = (f["cpr_top"] - f["cpr_bot"]) / f["c"] < 0.004
    return nz(cross_up(f["c"], f["cpr_top"])) & narrow & (f["k"] > 0), mn(f["cpr_bot"], f["low3"]) - 0.1 * f["atr"]


def pivot_r1(f):
    return nz(cross_up(f["c"], f["r1"])) & (f["k"] > 0), f["low3"] - 0.2 * f["atr"]


def donchian(f):
    return nz(cross_up(f["c"], f["dch"])), f["low6"] - 0.1 * f["atr"]


def keltner(f):
    return nz(cross_up(f["c"], f["kcu"])), f["ema20"]


def bb_squeeze(f):
    was = shift(f["squeeze"].astype(float), 1, 0) > 0
    return was & (f["c"] > f["bbu"]), f["bbm"]


def gap_and_go(f):
    first_h = np.where(f["k"] >= 1, _first(f, "h"), np.nan)
    first_l = _first(f, "l")
    green = _first(f, "c") > _first(f, "o")
    return nz(cross_up(f["c"], first_h)) & (f["gap"] > 0.4) & green & (f["k"] <= 6), first_l


def _first(f, key):
    out = np.where(f["first"], f[key], np.nan)
    return pd.Series(out).ffill().values


def open_drive(f):
    """First candle of the day is a strong one-way bar in the gap's direction: go with it at its close."""
    sig = (f["k"] == 0) & (f["c"] > f["o"]) & (f["body"] > 0.7 * f["rng"]) & (f["gap"] > -0.2) & (f["rng"] > f["atr"])
    return sig, f["l"]


def afternoon_hod(f):
    return (f["mins"] >= 810) & nz(cross_up(f["c"], f["hod_prev"])), f["low6"] - 0.1 * f["atr"]


def nr7_orb(f):
    s, st = orb15(f)
    return s & f["nr7"], st


def flag(f):
    """Pole (>2 ATR in 6 bars) then a tight 4-bar flag holding the top half of the pole, then a break of the flag."""
    c, h, l, atr = f["c"], f["h"], f["l"], f["atr"]
    fh = shift(pd.Series(h).rolling(4).max().values)
    fl = shift(pd.Series(l).rolling(4).min().values)
    pole_top, pole_bot = shift(c, 5), shift(c, 11)
    pole = pole_top - pole_bot
    ok = (pole > 2 * atr) & (fh - fl < 0.8 * atr) & (fl > pole_bot + 0.5 * pole)
    return nz(ok) & (c > fh), fl - 0.1 * atr


def inside_break(f):
    ph, pl = shift(f["h"]), shift(f["l"])
    return f["inside_prev"] & (f["c"] > shift(f["h"], 2)) & (f["c"] > ph), mn(pl, f["l"]) - 0.1 * f["atr"]


# ----------------------------------------------------------------------------- VWAP / averages / momentum
def vwap_pullback(f):
    touch = f["l"] <= f["vwap"] * 1.0008
    return touch & (f["c"] > f["vwap"]) & (f["c"] > f["o"]) & (f["k"] >= 3), mn(f["l"], f["low3"]) - 0.2 * f["atr"]


def vwap_reclaim(f):
    return nz(cross_up(f["c"], f["vwap"])) & (f["k"] >= 3), f["low3"] - 0.2 * f["atr"]


def ema_cross(f):
    return nz(cross_up(f["ema9"], f["ema21"])), f["low6"] - 0.1 * f["atr"]


def ema20_pullback(f):
    return (f["l"] <= f["ema20"]) & (f["c"] > f["ema20"]) & (f["c"] > f["o"]) & (f["ema20"] > f["ema50"]), f["low3"] - 0.2 * f["atr"]


def supertrend_flip(f):
    return (f["st_dir"] > 0) & (shift(f["st_dir"]) < 0), f["st_line"]


def macd_cross(f):
    return nz(cross_up(f["macd"], f["macds"])) & (f["c"] > f["ema50"]), f["low6"] - 0.1 * f["atr"]


def rsi_pullback(f):
    low_rsi = pd.Series(f["rsi"]).rolling(6).min().values
    return nz(cross_up(f["rsi"], 50)) & (shift(low_rsi) < 40), f["low6"] - 0.1 * f["atr"]


def rsi2_dip(f):
    """Connors-style: a 2-period RSI washout inside an uptrend, bought as it turns."""
    return (shift(f["rsi2"]) < 10) & (f["c"] > f["o"]) & (f["c"] > f["ema50"]), f["c"] - 1.5 * f["atr"]


def adx_trend(f):
    strong = (f["adx"] > 25) & (f["pdi"] > f["mdi"])
    return strong & nz(cross_up(f["c"], f["ema9"])), f["low3"] - 0.2 * f["atr"]


def heikin_ashi(f):
    return f["ha_nowick_bull"] & ~f["ha_bull_prev"], f["low3"] - 0.2 * f["atr"]


def momentum_combo(f):
    """VWAP + EMA stack + RSI turning up through 50."""
    return (f["c"] > f["vwap"]) & (f["ema9"] > f["ema21"]) & nz(cross_up(f["rsi"], 50)), f["low6"] - 0.1 * f["atr"]


def orb_vwap_rvol(f):
    s, st = orb15(f)
    return s & (f["c"] > f["vwap"]) & (np.nan_to_num(f["rvol"], nan=1.5) >= 1.5), st


def rs_pullback(f):
    """Stock leading its index (relative strength) bought on a pullback to the 20 EMA."""
    s, st = ema20_pullback(f)
    return s & (f["rs"] > 0.5), st


# ----------------------------------------------------------------------------- candles
def engulf_ema(f):
    near = f["l"] <= np.maximum(f["ema20"], f["vwap"]) * 1.002
    return f["bull_engulf"] & near, mn(f["l"], shift(f["l"])) - 0.1 * f["atr"]


def hammer_vwap(f):
    return f["hammer"] & (f["l"] <= f["vwap"] * 1.001) & (f["c"] > f["vwap"]), f["l"] - 0.1 * f["atr"]


def soldiers(f):
    return f["soldiers"] & (f["c"] > f["vwap"]), f["low3"] - 0.1 * f["atr"]


def morning_star(f):
    return f["morning_star"], f["low3"] - 0.1 * f["atr"]


def marubozu(f):
    return f["marubozu_bull"] & (f["c"] > f["hod_prev"]), f["l"]


# ----------------------------------------------------------------------------- smart money concepts
def sweep_pdl(f):
    return (f["l"] < f["pdl"]) & (f["c"] > f["pdl"]) & (f["k"] > 0), f["l"] - 0.1 * f["atr"]


def sweep_orl(f):
    return (f["k"] >= 3) & (f["l"] < f["or15l"]) & (f["c"] > f["or15l"]), f["l"] - 0.1 * f["atr"]


def bos(f):
    """Break of structure: a decisive close through the last confirmed swing high."""
    return nz(cross_up(f["c"], f["swh"])) & (f["body"] > 0.6 * f["rng"]), f["swl"] - 0.1 * f["atr"]


def order_block(f):
    """After a BOS, wait for price to come back into the last down-candle before the impulse (the order block)
    and close back above it; stop under the block. Valid the same day, for 30 bars."""
    o, h, l, c, atr, day = f["o"], f["h"], f["l"], f["c"], f["atr"], f["day"]
    b, _ = bos(f)
    sig, stop = np.zeros(f["n"], bool), np.full(f["n"], np.nan)
    zone, born, used = None, -1, True
    for i in range(1, f["n"]):
        if zone and (day[i] != day[born] or i - born > 30 or c[i] < zone[1]):
            zone = None
        if b[i]:
            for j in range(i - 1, max(i - 12, 0), -1):
                if c[j] < o[j] and day[j] == day[i]:
                    zone, born, used = (h[j], l[j]), i, False
                    break
            continue
        if zone and not used and l[i] <= zone[0] and c[i] > zone[0]:
            sig[i], stop[i], used = True, zone[1] - 0.1 * atr[i], True
    return sig, stop


def fvg_retest(f):
    """A bullish fair-value gap (bar i low above bar i-2 high) left by a strong candle; buy the first dip into it
    that closes back above the gap; stop below the gap."""
    h, l, c, o, atr, day = f["h"], f["l"], f["c"], f["o"], f["atr"], f["day"]
    sig, stop = np.zeros(f["n"], bool), np.full(f["n"], np.nan)
    gap, born = None, -1
    for i in range(2, f["n"]):
        if gap and (day[i] != day[born] or i - born > 24 or c[i] < gap[0]):
            gap = None
        if l[i] > h[i - 2] and (c[i - 1] - o[i - 1]) > 0.8 * atr[i] and day[i - 2] == day[i]:
            gap, born = (h[i - 2], l[i]), i
            continue
        if gap and i > born + 1 and l[i] <= gap[1] and c[i] > gap[1] and c[i] > o[i]:
            sig[i], stop[i] = True, gap[0] - 0.1 * atr[i]
            gap = None
    return sig, stop


def bos_fib(f):
    """BOS, then the pullback into the 50-62% zone of the impulse leg, entered on a bullish close."""
    h, l, c, o, atr, day = f["h"], f["l"], f["c"], f["o"], f["atr"], f["day"]
    b, _ = bos(f)
    swl = f["swl"]
    sig, stop = np.zeros(f["n"], bool), np.full(f["n"], np.nan)
    leg, born = None, -1
    for i in range(1, f["n"]):
        if leg:
            top = max(leg[1], h[i])
            leg = (leg[0], top)
            if day[i] != day[born] or i - born > 30 or l[i] < leg[0]:
                leg = None
        if b[i] and np.isfinite(swl[i]):
            leg, born = (swl[i], h[i]), i
            continue
        if leg:
            lo, hi = leg
            mid = hi - 0.5 * (hi - lo)
            if l[i] <= mid and c[i] > o[i] and c[i] > hi - 0.62 * (hi - lo):
                sig[i], stop[i] = True, lo - 0.1 * atr[i]
                leg = None
    return sig, stop


def double_bottom(f):
    """Two swing lows within 0.3 ATR, then a close through the high between them."""
    h, l, c, atr, day = f["h"], f["l"], f["c"], f["atr"], f["day"]
    swl, swh = f["swl"], f["swh"]
    sig, stop = np.zeros(f["n"], bool), np.full(f["n"], np.nan)
    lows = []
    for i in range(1, f["n"]):
        if np.isfinite(swl[i]) and swl[i] != swl[i - 1]:
            lows = [x for x in lows if day[x[1]] == day[i]][-1:] + [(swl[i], i)]
        if len(lows) == 2 and abs(lows[0][0] - lows[1][0]) < 0.3 * atr[i] and np.isfinite(swh[i]) \
                and c[i] > swh[i] and c[i - 1] <= swh[i] and swh[i] > max(lows[0][0], lows[1][0]) + 0.5 * atr[i]:
            sig[i], stop[i] = True, min(lows[0][0], lows[1][0]) - 0.1 * atr[i]
            lows = []
    return sig, stop


STRATEGIES = {
    # breakouts / opening
    "ORB 15m": orb15, "ORB 30m": orb30, "Gap & go": gap_and_go, "Opening drive": open_drive,
    "PDH breakout": pdh_break, "CPR breakout": cpr_break, "Pivot R1 break": pivot_r1, "Donchian 20": donchian,
    "Keltner break": keltner, "BB squeeze": bb_squeeze, "NR7 + ORB": nr7_orb, "Bull flag": flag,
    "Inside bar break": inside_break, "Afternoon HOD": afternoon_hod,
    # VWAP / MAs / momentum
    "VWAP pullback": vwap_pullback, "VWAP reclaim": vwap_reclaim, "EMA 9/21 cross": ema_cross,
    "EMA20 pullback": ema20_pullback, "Supertrend flip": supertrend_flip, "MACD cross": macd_cross,
    "RSI pullback": rsi_pullback, "RSI2 dip": rsi2_dip, "ADX trend": adx_trend, "Heikin-Ashi": heikin_ashi,
    "VWAP+EMA+RSI": momentum_combo, "ORB+VWAP+RVOL": orb_vwap_rvol, "RS pullback": rs_pullback,
    # candles
    "Engulfing @EMA": engulf_ema, "Hammer @VWAP": hammer_vwap, "3 soldiers": soldiers,
    "Morning star": morning_star, "Marubozu HOD": marubozu,
    # smart money
    "PDL sweep": sweep_pdl, "OR-low sweep": sweep_orl, "BOS": bos, "Order block": order_block,
    "FVG retest": fvg_retest, "BOS 50% pullback": bos_fib, "Double bottom": double_bottom,
}
