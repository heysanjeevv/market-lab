import numpy as np, pandas as pd
from lab import bars as B

def ew_index():
    cl = {}
    for p in sorted((B.ROOT / "1d").glob("*.pkl")):
        if p.stem in B.INDICES: continue
        d = pd.read_pickle(p); d = d[d.volume > 0]; d = d[~d.index.duplicated(keep="last")]
        cl[p.stem] = pd.Series(d.close.values, index=[x.date() for x in d.index])
    C = pd.DataFrame(cl).sort_index()
    r = C.pct_change(fill_method=None)
    r = r.where(r.abs() < 0.35)
    ew = (1 + r.mean(axis=1).fillna(0)).cumprod()
    return ew, C

def table(t, ew, cut="2024-07-01"):
    t = t.copy()
    t["d_in"] = pd.to_datetime(t.d_in); t["d_out"] = pd.to_datetime(t.d_out)
    e = pd.Series(ew.values, index=pd.to_datetime(ew.index))
    # benchmark: equal-weight universe from entry open (~prev close) to exit
    t["bench"] = (e.reindex(t.d_out, method="ffill").values / e.reindex(t.d_in - pd.Timedelta(days=1), method="ffill").values - 1) * 100
    t["excess"] = t.gross - t.bench
    t["yr"] = t.d_in.dt.year
    c = pd.Timestamp(cut)
    g = t.groupby(["strat", "exit"])
    out = pd.DataFrame({"n": g.size(), "win%": g.net.apply(lambda x: (x > 0).mean() * 100).round(1), "net%": g.net.mean().round(2),
                        "R": g.R.mean().round(3), "days": g.days.mean().round(1), "excess%": g.excess.mean().round(2),
                        "IS_net": t[t.d_in < c].groupby(["strat", "exit"]).net.mean().round(2),
                        "OOS_net": t[t.d_in >= c].groupby(["strat", "exit"]).net.mean().round(2),
                        "OOS_exc": t[t.d_in >= c].groupby(["strat", "exit"]).excess.mean().round(2)})
    yrs = t.groupby(["strat", "exit", "yr"]).net.mean().unstack().round(2)
    out["yrs+"] = (yrs > 0).sum(axis=1)
    return out.join(yrs), t
