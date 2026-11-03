"""Walk-forward strategy rotation: every month, rank all combos by how they did over the trailing window
(net of costs) and trade only the top few next month. Only past data decides; the score is the chain of
next-month results."""
import numpy as np
import pandas as pd

from .analyze import FILTERS
from .engine import MODES

USE_FILTERS = ["all", "cost<0.2R", "leaders", "index-aligned", "movers", "indices"]


def monthly(t, filters=USE_FILTERS, modes=MODES):
    t = t.copy()
    t["month"] = pd.to_datetime(t.ts, unit="s").dt.to_period("M")
    cols = {}
    cnt = {}
    for fn in filters:
        g = t[FILTERS[fn](t)]
        for m in modes:
            net = g[m] - g.cost_r
            k = net.groupby([g.strat, g.trend, g.month]).agg(["sum", "count"])
            for (s, tr), sub in k.groupby(level=[0, 1]):
                key = (s, tr, m, fn)
                cols[key] = sub["sum"].droplevel([0, 1])
                cnt[key] = sub["count"].droplevel([0, 1])
    S = pd.DataFrame(cols).fillna(0).sort_index()
    N = pd.DataFrame(cnt).fillna(0).sort_index()
    return S, N


def rotate(S, N, look=6, top=5, min_n=40, min_avg=0.05):
    months = S.index
    res = []
    for i in range(look, len(months)):
        past_s, past_n = S.iloc[i - look:i].sum(), N.iloc[i - look:i].sum()
        avg = (past_s / past_n.replace(0, np.nan))
        # also require the latest half of the window to be positive (still working now)
        recent = S.iloc[i - look // 2:i].sum()
        ok = (past_n >= min_n) & (avg > min_avg) & (recent > 0)
        pick = avg[ok].sort_values(ascending=False).head(top)
        r = S.iloc[i][pick.index].sum() if len(pick) else 0.0
        n = N.iloc[i][pick.index].sum() if len(pick) else 0
        res.append({"month": str(months[i]), "picked": len(pick), "trades": int(n), "R": round(float(r), 1),
                    "avgR": round(float(r / n), 3) if n else 0, "names": [" / ".join(k) for k in pick.index[:3]]})
    return pd.DataFrame(res)
