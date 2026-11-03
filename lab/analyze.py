"""Score every (strategy, trend filter, exit rule, stock filter) on the trade log, with an in-sample /
out-of-sample split by date so a winner has to keep working on days it was not chosen on."""
import numpy as np
import pandas as pd

from .engine import MODES, INDEX_SYMS

FILTERS = {
    "all": lambda t: np.ones(len(t), bool),
    "cost<0.2R": lambda t: t.cost_r < 0.2,
    "cost<0.12R": lambda t: t.cost_r < 0.12,
    "movers": lambda t: (t.chg > 1.0) & (t.cost_r < 0.2),                       # already up 1%+ in the trade's direction
    "rvol>1.5": lambda t: (t.rvol > 1.5) & (t.cost_r < 0.2),
    "leaders": lambda t: (t.rs > 0.5) & (t.cost_r < 0.2),                      # beating the index
    "index-aligned": lambda t: t.ix_al & (t.cost_r < 0.2),
    "in-play": lambda t: (t.rvol > 1.5) & (t.chg > 1.0) & t.ix_al & (t.cost_r < 0.2),
    "indices": lambda t: t.sym.isin(INDEX_SYMS),
    "indices cost<0.2R": lambda t: t.sym.isin(INDEX_SYMS) & (t.cost_r < 0.2),
}


def stats(r, ts=None):
    r = np.asarray(r, float)
    if len(r) == 0:
        return {"n": 0}
    w = r[r > 0].sum()
    l = -r[r < 0].sum()
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    return {"n": len(r), "win%": round((r > 0).mean() * 100, 1), "avgR": round(r.mean(), 3),
            "PF": round(w / l, 2) if l > 0 else 99, "totR": round(r.sum(), 1), "maxDD_R": round(dd, 1)}


def score(t, split_frac=0.67, min_n=60):
    t = t.sort_values("ts")
    days = np.sort(t.ts.unique())
    cut = days[int(len(days) * split_frac)] if len(days) else 0
    out = []
    for fname, fn in FILTERS.items():
        tf = t[fn(t)]
        for (strat, trend), g in tf.groupby(["strat", "trend"]):
            for m in MODES:
                net = g[m] - g.cost_r
                ins, oos = net[g.ts < cut], net[g.ts >= cut]
                if len(ins) < min_n * split_frac or len(oos) < min_n * (1 - split_frac):
                    continue
                a, b = stats(ins), stats(oos)
                # day-level consistency: share of trading days that ended positive
                daily = net.groupby(pd.to_datetime(g.ts, unit="s").dt.date).sum()
                out.append({"strat": strat, "trend": trend, "exit": m, "filter": fname,
                            "n": len(net), "avgR": round(net.mean(), 3), "win%": round((net > 0).mean() * 100, 1),
                            "PF": stats(net)["PF"], "IS_avgR": a["avgR"], "OOS_avgR": b["avgR"], "OOS_PF": b["PF"],
                            "OOS_n": b["n"], "day+%": round((daily > 0).mean() * 100, 1), "maxDD_R": stats(net)["maxDD_R"],
                            "costR": round(g.cost_r.mean(), 3), "grossR": round(g[m].mean(), 3)})
    return pd.DataFrame(out)
