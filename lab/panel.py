"""Cross-sectional tests: every day, at a fixed decision time, rank the whole universe and trade the few
strongest stocks in the index's direction (the 'most moving / trending stocks' idea), plus the
time-series version on the indices themselves. Works on the wide panel of all symbols."""
import numpy as np
import pandas as pd

from . import bars as B
from .engine import cost_frac, INDEX_SYMS


def load_panel(tf, syms=None):
    syms = syms or sorted(p.stem for p in (B.ROOT / tf).glob("*.pkl"))
    fr = {s: B.load(s, tf) for s in syms}
    fr = {s: d for s, d in fr.items() if d is not None and len(d) > 200}
    idx = sorted(set().union(*[d.index for d in fr.values()]))
    idx = pd.DatetimeIndex(idx)
    P = {k: pd.DataFrame({s: d[k] for s, d in fr.items()}).reindex(idx) for k in ("open", "high", "low", "close", "volume")}
    daily = {s: B.load(s, "1d") for s in fr}
    return P, daily


def daily_ctx(daily, dates):
    """Yesterday's trend (close>EMA20>EMA50 -> +1, mirror -> -1) and daily ATR% per symbol, per date."""
    tr, atr, rng = {}, {}, {}
    for s, d in daily.items():
        if d is None or len(d) < 60:
            continue
        c = d["close"]
        e20, e50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
        t = np.where((c > e20) & (e20 > e50), 1, np.where((c < e20) & (e20 < e50), -1, 0))
        dtr = pd.concat([d.high - d.low, (d.high - c.shift()).abs(), (d.low - c.shift()).abs()], axis=1).max(axis=1)
        a = dtr.ewm(alpha=1 / 14, adjust=False).mean() / c * 100
        ix = [x.date() for x in d.index]
        tr[s] = pd.Series(t, index=ix).groupby(level=0).last().shift(1)
        atr[s] = pd.Series(a.values, index=ix).groupby(level=0).last().shift(1)
    return pd.DataFrame(tr).reindex(dates), pd.DataFrame(atr).reindex(dates)


def simulate(P, daily, k_dec, top=5, rank="chg", stop_mode="lod", stop_atr=0.6, exit_mode="eod", rr=None,
             min_rvol=0.0, need_trend=True, index_dir=True, universe="stocks", log_trades=False):
    O, H, L, C, V = (P[k] for k in ("open", "high", "low", "close", "volume"))
    days = pd.Series(O.index.date, index=O.index)
    dates = sorted(set(days))
    trend, atr = daily_ctx(daily, dates)
    nifty = C["NIFTY"]
    syms = [s for s in C.columns if (s in INDEX_SYMS) == (universe == "indices")]
    cost = {s: cost_frac(s) for s in syms}
    out = []
    prev_close = None
    by_day = {d: np.nonzero((days == d).values)[0] for d in dates}
    for d in dates:
        ii = by_day[d]
        if prev_close is None or len(ii) <= k_dec + 2:
            prev_close = C.iloc[ii[-1]] if len(ii) else prev_close
            continue
        kd, ke = ii[k_dec], ii[k_dec + 1]
        tf_min = int((O.index[ii[1]] - O.index[ii[0]]).total_seconds() // 60)
        mins = O.index[ii].hour * 60 + O.index[ii].minute
        last = ii[np.nonzero(mins + tf_min <= 915)[0][-1]]
        pc = prev_close
        chg = (C.iloc[kd] / pc - 1) * 100
        day_open = O.iloc[ii[0]]
        from_open = (C.iloc[kd] / day_open - 1) * 100
        nchg = chg.get("NIFTY", np.nan)
        side_ix = 1 if (nifty.iloc[kd] > O["NIFTY"].iloc[ii[0]] and nchg > 0) else -1 if (nifty.iloc[kd] < O["NIFTY"].iloc[ii[0]] and nchg < 0) else 0
        cv = V.iloc[ii[0]:kd + 1].sum()
        prev_close = C.iloc[ii[-1]]
        cand = []
        for s in syms:
            c0 = chg.get(s)
            if not np.isfinite(c0) or not np.isfinite(O[s].iloc[ke]):
                continue
            side = int(np.sign(c0)) if not index_dir else side_ix
            if side == 0 or np.sign(c0) != side or np.sign(from_open[s]) != side:
                continue
            if need_trend and (trend.at[d, s] if s in trend.columns else np.nan) != side:
                continue
            a = atr.at[d, s] if s in atr.columns else np.nan
            if not np.isfinite(a):
                continue
            score = {"chg": side * c0, "rs": side * (c0 - nchg), "atrmove": side * c0 / a, "open": side * from_open[s]}[rank]
            cand.append((score, s, side, a))
        cand.sort(reverse=True)
        for score, s, side, a in cand[:top]:
            e = O[s].iloc[ke]
            hs, ls, cs = H[s].iloc[ke:last + 1].values, L[s].iloc[ke:last + 1].values, C[s].iloc[ke:last + 1].values
            if side < 0:
                e, hs, ls, cs = -e, -ls, -hs, -cs
            lod = (L[s].iloc[ii[0]:kd + 1].min() if side > 0 else -H[s].iloc[ii[0]:kd + 1].max())
            stop = {"lod": lod, "atr": e - stop_atr * a / 100 * abs(e)}[stop_mode]
            r = e - stop
            if not np.isfinite(r) or r <= 0 or np.isnan(cs).all():
                continue
            hit = ls <= stop
            ks = int(np.argmax(hit)) if hit.any() else len(ls)
            if rr:
                th = hs >= e + rr * r
                kt = int(np.argmax(th)) if th.any() else len(hs)
            else:
                kt = len(hs)
            last_c = cs[~np.isnan(cs)][-1]
            if ks <= kt and ks < len(ls):
                R, xi = -1.0, ks
            elif kt < len(hs):
                R, xi = rr, kt
            else:
                R, xi = (last_c - e) / r, len(cs) - 1
            cr = cost[s] * abs(e) / r
            out.append({"date": d, "sym": s, "side": side, "score": score, "R": R - cr, "gross": R, "cost_r": cr,
                        "pct": (R * r / abs(e) - cost[s]) * 100, "risk_pct": r / abs(e) * 100,
                        "t_in": int(O.index[ke].timestamp()), "t_out": int(O.index[ke + xi].timestamp()),
                        "entry": abs(e), "stop": abs(stop), "target": abs(e + rr * r) if rr else None})
    return pd.DataFrame(out)


def summary(t, split=0.67):
    if t is None or not len(t):
        return {"n": 0}
    t = t.sort_values("date")
    day = t.groupby("date")["pct"].mean()            # equal capital per position, per day
    dR = t.groupby("date")["R"].sum()
    cut = day.index[int(len(day) * split)]
    eq = day.cumsum()
    dd = (eq.cummax() - eq).max()
    yr = pd.Series(day.values, index=pd.to_datetime(day.index)).groupby(lambda x: x.year).sum().round(1).to_dict()
    return {"n": len(t), "days": len(day), "win%": round((t.R > 0).mean() * 100, 1), "avgR": round(t.R.mean(), 3),
            "grossR": round(t.gross.mean(), 3), "costR": round(t.cost_r.mean(), 3),
            "PF": round(t.R[t.R > 0].sum() / max(-t.R[t.R < 0].sum(), 1e-9), 2),
            "IS_avgR": round(t.R[t.date < cut].mean(), 3), "OOS_avgR": round(t.R[t.date >= cut].mean(), 3),
            "day+%": round((dR > 0).mean() * 100, 1), "sum_day%": round(day.sum(), 1), "maxDD%": round(dd, 1), "years": yr}
