"""Gap study on the intraday bars themselves (same source for yesterday's close and today's open, so
splits and Yahoo's daily-open quirks cannot fake a gap)."""
import numpy as np
import pandas as pd

from . import bars as B
from .engine import cost_frac


def day_table(sym, tf="60m"):
    d = B.load(sym, tf)
    if d is None or len(d) < 200:
        return None
    d = d.copy()
    d["date"] = d.index.date
    g = d.groupby("date")
    t = pd.DataFrame({"o": g.open.first(), "c": g.close.last(), "h": g.high.max(), "l": g.low.min(), "n": g.size()})
    t["pc"] = t.c.shift()
    t["gap"] = (t.o / t.pc - 1) * 100
    tr = pd.concat([t.h - t.l, (t.h - t.pc).abs(), (t.l - t.pc).abs()], axis=1).max(axis=1)
    t["atr"] = (tr.ewm(alpha=1 / 14, adjust=False).mean() / t.c * 100).shift()
    e20, e50 = t.c.ewm(span=20, adjust=False).mean(), t.c.ewm(span=50, adjust=False).mean()
    t["trend"] = np.where((t.c > e20) & (e20 > e50), 1, np.where((t.c < e20) & (e20 < e50), -1, 0))
    t["trend"] = t["trend"].shift()
    t["vol20"] = g.volume.sum().rolling(20).mean().shift()
    return t, d


def trade_day(d_day, side, entry_k, stop_pct, tgt_pct, exit_mins=915, tf=60):
    """Long (side=1) or short from bar entry_k's open; stop/target in % of entry; exit at the last bar
    closing by 15:15. Stop is checked before target inside a bar (conservative)."""
    o, h, l, c = (d_day[k].values for k in ("open", "high", "low", "close"))
    mins = d_day.index.hour * 60 + d_day.index.minute
    if len(o) <= entry_k:
        return None
    e = o[entry_k]
    stop = e * (1 - side * stop_pct / 100)
    tgt = e * (1 + side * tgt_pct / 100) if tgt_pct else None
    last = np.nonzero(mins + tf <= exit_mins)[0]
    last = last[-1] if len(last) else len(o) - 1
    for j in range(entry_k, last + 1):
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            return -stop_pct, j
        if tgt and ((side > 0 and h[j] >= tgt) or (side < 0 and l[j] <= tgt)):
            return tgt_pct, j
    return side * (c[last] / e - 1) * 100, last
