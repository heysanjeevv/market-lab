"""Portfolio simulation for swing systems: Rs 10 lakh, at most K positions of equal size, signals taken in order
of relative strength when a slot is free, marked to market every day on closing prices, delivery costs on
every round trip. Benchmarks: Nifty 50 and the equal-weight universe (same survivorship bias as the tests)."""
import numpy as np
import pandas as pd

from . import bars as B
from . import swing as S


def panels():
    O, C = {}, {}
    for p in sorted((B.ROOT / "1d").glob("*.pkl")):
        if p.stem in B.INDICES:
            continue
        d = pd.read_pickle(p)
        d = d[d.volume > 0]
        d = d[~d.index.duplicated(keep="last")]
        idx = [x.date() for x in d.index]
        O[p.stem] = pd.Series(d.open.values, index=idx)
        C[p.stem] = pd.Series(d.close.values, index=idx)
    return pd.DataFrame(O).sort_index(), pd.DataFrame(C).sort_index()


def simulate(trades, O, C, k=10, capital=1_000_000, start=None, mkt=None):
    """trades: rows with sym, d_in, d_out, gross (entry at d_in open, exit price implied by gross)."""
    t = trades.sort_values(["d_in", "rs63"], ascending=[True, False])
    if start is not None:
        t = t[pd.to_datetime(t.d_in) >= pd.Timestamp(start)]
    by_day = {d: g for d, g in t.groupby("d_in")}
    dates = [d for d in C.index if (start is None or pd.Timestamp(d) >= pd.Timestamp(start))]
    cash, pos, eq, log = capital, {}, [], []
    for d in dates:
        # exits first (at the trade's own exit price, after costs)
        for sym in [s for s, p in pos.items() if p["d_out"] <= d]:
            p = pos.pop(sym)
            px = p["entry"] * (1 + p["gross"] / 100)
            cash += p["sh"] * px - p["sh"] * p["entry"] * S.COST
            log.append({**p, "pnl": p["sh"] * (px - p["entry"]) - p["sh"] * p["entry"] * S.COST})
        if d in by_day and (mkt is None or mkt.get(d, True)):
            value = cash + sum(p["sh"] * C.at[d, s] if np.isfinite(C.at[d, s]) else p["sh"] * p["entry"] for s, p in pos.items())
            for _, r in by_day[d].iterrows():
                if len(pos) >= k or r.sym in pos:
                    continue
                e = O.at[d, r.sym] if r.sym in O.columns else np.nan
                if not np.isfinite(e) or e <= 0:
                    continue
                alloc = min(value / k, cash)
                sh = int(alloc // e)
                if sh <= 0:
                    continue
                cash -= sh * e
                pos[r.sym] = {"sym": r.sym, "sh": sh, "entry": e, "d_in": d, "d_out": r.d_out, "gross": r.gross, "strat": r.strat}
        mtm = cash + sum(p["sh"] * (C.at[d, s] if np.isfinite(C.at[d, s]) else p["entry"]) for s, p in pos.items())
        eq.append((d, mtm))
    e = pd.Series(dict(eq))
    return e, pd.DataFrame(log)


def metrics(e):
    e = e.dropna()
    yrs = (pd.Timestamp(e.index[-1]) - pd.Timestamp(e.index[0])).days / 365.25
    r = e.pct_change().dropna()
    dd = (e / e.cummax() - 1).min() * 100
    by_year = e.groupby(pd.to_datetime(e.index).year).agg(["first", "last"])
    return {"CAGR%": round(((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1) * 100, 1), "maxDD%": round(dd, 1),
            "Sharpe": round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() else 0,
            "years%": {int(y): round((v["last"] / v["first"] - 1) * 100, 1) for y, v in by_year.iterrows()}}


def bench(C, start):
    n = B.load("NIFTY", "1d").close
    n = pd.Series(n.values, index=[x.date() for x in n.index])
    n = n[[pd.Timestamp(x) >= pd.Timestamp(start) for x in n.index]]
    r = C.pct_change(fill_method=None)
    r = r.where(r.abs() < 0.35).mean(axis=1).fillna(0)
    ew = (1 + r[[pd.Timestamp(x) >= pd.Timestamp(start) for x in r.index]]).cumprod()
    return n / n.iloc[0] * 1e6, ew * 1e6


def momentum(C, start, top=20, every=21, regime=True):
    """Classic cross-sectional momentum: each month hold the top N by 6-month return (skipping the last week),
    only while Nifty is above its 200-day average; costs on every change."""
    n = B.load("NIFTY", "1d").close
    n = pd.Series(n.values, index=[x.date() for x in n.index])
    up = (n > n.rolling(200).mean()).reindex(C.index).ffill().fillna(False)
    R = C.pct_change(fill_method=None)
    R = R.where(R.abs() < 0.35)
    score = C.shift(5) / C.shift(126) - 1
    dates = [d for d in C.index if pd.Timestamp(d) >= pd.Timestamp(start)]
    hold, val, out = [], 1e6, []
    for i, d in enumerate(dates):
        if i:
            val *= 1 + (R.loc[d, hold].mean() if hold else 0)
        if i % every == 0:
            new = list(score.loc[d].dropna().sort_values(ascending=False).index[:top]) if (up.get(d, True) or not regime) else []
            changed = len(set(new) ^ set(hold)) / max(top, 1) / 2 if (new or hold) else 0
            val *= 1 - changed * S.COST
            hold = new
        out.append((d, val))
    return pd.Series(dict(out))
