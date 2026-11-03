"""Every indicator the strategies need, computed bar by bar with no look-ahead: a value at bar i only
uses bars up to and including i (signals are acted on at the NEXT bar's open)."""
import numpy as np
import pandas as pd


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def rma(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().values


def shift(x, k=1, fill=np.nan):
    out = np.empty_like(x, dtype=float)
    if k > 0:
        out[:k] = fill
        out[k:] = x[:-k]
    else:
        out[:] = x
    return out


def group_cum(x, day, fn):
    return pd.Series(x).groupby(day).transform(fn).values


def supertrend(h, l, c, atr, mult=3.0):
    hl2 = (h + l) / 2
    up, dn = hl2 - mult * atr, hl2 + mult * atr
    n = len(c)
    fu, fd, d = up.copy(), dn.copy(), np.ones(n)
    for i in range(1, n):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        d[i] = 1 if c[i] > fd[i - 1] else -1 if c[i] < fu[i - 1] else d[i - 1]
    return d, np.where(d > 0, fu, fd)


def swings(h, l, k=2):
    """Fractal swing highs/lows, CONFIRMED k bars later: arrays hold the last confirmed level at each bar."""
    n = len(h)
    sh, sl = np.full(n, np.nan), np.full(n, np.nan)
    last_h = last_l = np.nan
    for i in range(2 * k, n):
        j = i - k
        if h[j] == h[j - k:i + 1].max():
            last_h = h[j]
        if l[j] == l[j - k:i + 1].min():
            last_l = l[j]
        sh[i], sl[i] = last_h, last_l
    return sh, sl


def mirror(df):
    """Upside-down copy of a price frame: a SHORT setup on df is exactly a LONG setup on mirror(df), so every
    strategy is written once, for longs, and run on both."""
    m = pd.DataFrame({"open": -df["open"], "high": -df["low"], "low": -df["high"], "close": -df["close"], "volume": df["volume"]}, index=df.index)
    return m


def build(df, daily, tf_min=5, index_feat=None, sign=1):
    """df: intraday OHLCV (IST index, regular session), already mirrored when sign=-1. daily: daily bars (same
    orientation). Percent features carry `sign` so they read 'in the trade's direction'. Returns dict of arrays."""
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    v = df["volume"].values.astype(float)
    t = df.index
    date = np.array(t.date)
    day = pd.factorize(date)[0]
    first = np.r_[True, day[1:] != day[:-1]]
    last = np.r_[day[1:] != day[:-1], True]
    k = pd.Series(1, index=range(len(c))).groupby(day).cumsum().values - 1
    mins = t.hour.values * 60 + t.minute.values
    f = dict(o=o, h=h, l=l, c=c, v=v, day=day, k=k, first=first, last=last, mins=mins, n=len(c), sign=sign,
             ts=np.asarray(t.tz_convert("UTC").tz_localize(None).values.astype("datetime64[s]").astype("int64")), tf=tf_min)
    f["date"] = date

    tr = np.maximum(h - l, np.maximum(abs(h - shift(c, 1, c[0])), abs(l - shift(c, 1, c[0]))))
    f["atr"] = rma(tr, 14)
    for n in (9, 20, 21, 50, 200):
        f[f"ema{n}"] = ema(c, n)
    # VWAP per day (indices have no volume: equal weights)
    w = np.where(v > 0, v, 1.0)
    tp = (h + l + c) / 3
    f["vwap"] = group_cum(tp * w, day, "cumsum") / group_cum(w, day, "cumsum")
    # RSI
    d = np.diff(c, prepend=c[0])
    f["rsi"] = 100 - 100 / (1 + rma(np.maximum(d, 0), 14) / np.maximum(rma(np.maximum(-d, 0), 14), 1e-12))
    f["rsi2"] = 100 - 100 / (1 + rma(np.maximum(d, 0), 2) / np.maximum(rma(np.maximum(-d, 0), 2), 1e-12))
    # MACD
    m = ema(c, 12) - ema(c, 26)
    f["macd"], f["macds"] = m, ema(m, 9)
    # Bollinger / Keltner
    sma = pd.Series(c).rolling(20).mean().values
    sd = pd.Series(c).rolling(20).std().values
    f["bbu"], f["bbl"], f["bbm"] = sma + 2 * sd, sma - 2 * sd, sma
    f["bbw"] = abs(4 * sd / sma)
    f["kcu"], f["kcl"] = f["ema20"] + 1.5 * f["atr"], f["ema20"] - 1.5 * f["atr"]
    f["squeeze"] = (f["bbu"] < f["kcu"]) & (f["bbl"] > f["kcl"])
    # Supertrend
    f["st_dir"], f["st_line"] = supertrend(h, l, c, rma(tr, 10), 3.0)
    # ADX
    up, dn = h - shift(h, 1, h[0]), shift(l, 1, l[0]) - l
    pdm, mdm = np.where((up > dn) & (up > 0), up, 0), np.where((dn > up) & (dn > 0), dn, 0)
    atr14 = np.maximum(f["atr"], 1e-12)
    pdi, mdi = 100 * rma(pdm, 14) / atr14, 100 * rma(mdm, 14) / atr14
    f["pdi"], f["mdi"] = pdi, mdi
    f["adx"] = rma(100 * abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12), 14)
    # Donchian (previous 20 bars)
    f["dch"] = shift(pd.Series(h).rolling(20).max().values)
    f["dcl"] = shift(pd.Series(l).rolling(20).min().values)
    # Heikin-Ashi
    hac = (o + h + l + c) / 4
    hao = pd.Series(shift(hac, 1, o[0])).ewm(alpha=0.5, adjust=False).mean().values
    f["ha_bull"], f["ha_bear"] = hac > hao, hac < hao
    f["ha_nowick_bull"] = (hac > hao) & (hao <= l + 1e-9)        # bullish HA candle with no lower wick
    f["ha_bull_prev"] = shift(f["ha_bull"].astype(float), 1, 0) > 0

    # Day structure
    f["dopen"] = group_cum(o, day, "first")
    f["hod"] = group_cum(h, day, "cummax")            # includes current bar
    f["lod"] = group_cum(l, day, "cummin")
    f["hod_prev"] = np.where(first, np.nan, shift(f["hod"]))
    f["lod_prev"] = np.where(first, np.nan, shift(f["lod"]))
    nb15, nb30 = max(1, 15 // tf_min), max(1, 30 // tf_min)
    for name, nb in (("or15", nb15), ("or30", nb30)):
        hh = pd.Series(np.where(k < nb, h, np.nan)).groupby(day).transform("max").values
        ll = pd.Series(np.where(k < nb, l, np.nan)).groupby(day).transform("min").values
        f[name + "h"], f[name + "l"] = np.where(k >= nb, hh, np.nan), np.where(k >= nb, ll, np.nan)   # known only after
    # previous day (from intraday bars)
    dd = pd.DataFrame({"h": h, "l": l, "c": c, "day": day}).groupby("day").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    for col in ("h", "l", "c"):
        f["pd" + col] = dd[col].shift(1).reindex(day).values
    P = (f["pdh"] + f["pdl"] + f["pdc"]) / 3
    BC = (f["pdh"] + f["pdl"]) / 2
    f["cpr_top"], f["cpr_bot"] = np.maximum(BC, 2 * P - BC), np.minimum(BC, 2 * P - BC)
    f["r1"], f["s1"] = 2 * P - f["pdl"], 2 * P - f["pdh"]
    f["gap"] = sign * (f["dopen"] / f["pdc"] - 1) * 100
    f["chg"] = sign * (c / f["pdc"] - 1) * 100

    # daily context (yesterday's values only)
    if daily is not None and len(daily) > 60:
        dc = daily["close"]
        e20, e50 = dc.ewm(span=20, adjust=False).mean(), dc.ewm(span=50, adjust=False).mean()
        dtr = pd.concat([daily["high"] - daily["low"], (daily["high"] - dc.shift()).abs(), (daily["low"] - dc.shift()).abs()], axis=1).max(axis=1)
        datr = dtr.ewm(alpha=1 / 14, adjust=False).mean()
        ctx = pd.DataFrame({"dup": (dc > e20) & (e20 > e50), "ddn": (dc < e20) & (e20 < e50),
                            "datr_pct": (datr / dc * 100).abs(), "d_e20": e20, "d_ret5": sign * dc.pct_change(5) * 100,
                            "nr7": (daily["high"] - daily["low"]) <= (daily["high"] - daily["low"]).rolling(7).min()}).shift(1)
        ctx.index = [x.date() for x in ctx.index.tz_convert("Asia/Kolkata")] if ctx.index.tz is not None else [x.date() for x in ctx.index]
        ctx = ctx[~ctx.index.duplicated(keep="last")]
        r = ctx.reindex(date)
        f["dup"] = r["dup"].fillna(False).values.astype(bool)
        f["ddn"] = r["ddn"].fillna(False).values.astype(bool)
        f["datr_pct"] = r["datr_pct"].values.astype(float)
        f["d_ret5"] = r["d_ret5"].values.astype(float)
        f["nr7"] = r["nr7"].fillna(False).values.astype(bool)
    else:
        f["dup"] = f["ddn"] = f["nr7"] = np.zeros(len(c), bool)
        f["datr_pct"] = f["d_ret5"] = np.full(len(c), np.nan)

    # relative volume: today's cumulative volume vs the average of the last 10 days at the same bar
    cv = group_cum(v, day, "cumsum")
    piv = pd.DataFrame({"day": day, "k": k, "cv": cv}).pivot(index="day", columns="k", values="cv")
    avg = piv.rolling(10, min_periods=5).mean().shift(1)
    f["rvol"] = np.array([avg.at[dd_, kk] if kk in avg.columns else np.nan for dd_, kk in zip(day, k)]) if len(piv) else np.full(len(c), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["rvol"] = cv / f["rvol"]
    f["vavg"] = pd.Series(v).rolling(20).mean().values

    # candles
    body, rng = abs(c - o), np.maximum(h - l, 1e-12)
    uw, lw = h - np.maximum(o, c), np.minimum(o, c) - l
    po, pc, ph, pl = shift(o), shift(c), shift(h), shift(l)
    f["bull_engulf"] = (c > o) & (pc < po) & (c >= po) & (o <= pc) & (body > 0.5 * rng)
    f["bear_engulf"] = (c < o) & (pc > po) & (c <= po) & (o >= pc) & (body > 0.5 * rng)
    f["hammer"] = (lw >= 2 * body) & (uw <= 0.25 * rng) & (rng > 0.5 * f["atr"])
    f["star"] = (uw >= 2 * body) & (lw <= 0.25 * rng) & (rng > 0.5 * f["atr"])
    f["inside"] = (h < ph) & (l > pl)
    f["inside_prev"] = shift(f["inside"].astype(float), 1, 0) > 0
    f["marubozu_bull"] = (c > o) & (body >= 0.8 * rng) & (rng > 1.2 * f["atr"])
    f["marubozu_bear"] = (c < o) & (body >= 0.8 * rng) & (rng > 1.2 * f["atr"])
    bull3 = (c > o) & (shift(c) > shift(o)) & (shift(c, 2) > shift(o, 2)) & (c > shift(c)) & (shift(c) > shift(c, 2))
    bear3 = (c < o) & (shift(c) < shift(o)) & (shift(c, 2) < shift(o, 2)) & (c < shift(c)) & (shift(c) < shift(c, 2))
    f["soldiers"], f["crows"] = bull3, bear3
    # morning / evening star: big down, small body, big up closing above the first's midpoint
    b2, b1 = abs(shift(c, 2) - shift(o, 2)), abs(pc - po)
    f["morning_star"] = (shift(c, 2) < shift(o, 2)) & (b2 > 0.6 * shift(f["atr"], 2)) & (b1 < 0.4 * b2) & (c > o) & (c > (shift(o, 2) + shift(c, 2)) / 2)
    f["evening_star"] = (shift(c, 2) > shift(o, 2)) & (b2 > 0.6 * shift(f["atr"], 2)) & (b1 < 0.4 * b2) & (c < o) & (c < (shift(o, 2) + shift(c, 2)) / 2)
    f["body"], f["rng"] = body, rng

    # structure (SMC)
    f["swh"], f["swl"] = swings(h, l, 2)
    f["low3"] = pd.Series(l).rolling(3).min().values
    f["high3"] = pd.Series(h).rolling(3).max().values
    f["low6"] = pd.Series(l).rolling(6).min().values
    f["high6"] = pd.Series(h).rolling(6).max().values

    # index context (aligned by timestamp)
    if index_feat is not None:
        s = pd.DataFrame(index_feat, index=index_feat["ts"])
        s = s[~s.index.duplicated(keep="last")].reindex(f["ts"])
        f["ix_up"] = s["up"].fillna(False).values.astype(bool)
        f["ix_dn"] = s["dn"].fillna(False).values.astype(bool)
        f["ix_chg"] = s["chg"].values.astype(float)
        f["rs"] = f["chg"] - f["ix_chg"]
    else:
        f["ix_up"] = f["ix_dn"] = np.ones(len(c), bool)
        f["ix_chg"] = np.zeros(len(c))
        f["rs"] = np.zeros(len(c))
    for key in ("bull_engulf", "bear_engulf", "hammer", "star", "inside", "marubozu_bull", "marubozu_bear",
                "soldiers", "crows", "morning_star", "evening_star", "squeeze"):
        f[key] = np.nan_to_num(f[key].astype(float)) > 0
    return f


def index_context(f):
    """What a stock strategy may know about the index at each bar."""
    return {"ts": f["ts"], "up": (f["c"] > f["vwap"]) & (f["ema21"] > f["ema50"]),
            "dn": (f["c"] < f["vwap"]) & (f["ema21"] < f["ema50"]), "chg": f["chg"]}
