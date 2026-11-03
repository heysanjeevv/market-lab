"""Market regime known BEFORE the session (yesterday's close): Nifty trend, Nifty trendiness (efficiency
ratio), India VIX level and its change, Nifty's recent range. Used to switch strategy families on/off."""
import numpy as np
import pandas as pd

from . import bars as B


def table():
    n = B.load("NIFTY", "1d")
    v = B.load("INDIAVIX", "1d")
    c = n.close
    idx = [x.date() for x in n.index]
    e20, e50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    er = (c - c.shift(10)).abs() / c.diff().abs().rolling(10).sum()
    rng = ((n.high - n.low) / c * 100).rolling(5).mean()
    vix = pd.Series(v.close.values, index=[x.date() for x in v.index]).groupby(level=0).last()
    t = pd.DataFrame({"n_up": ((c > e20) & (e20 > e50)).values, "n_dn": ((c < e20) & (e20 < e50)).values,
                      "er10": er.values, "rng5": rng.values, "ret5": (c.pct_change(5) * 100).values}, index=idx)
    t = t.groupby(level=0).last()
    t["vix"] = vix.reindex(t.index).ffill()
    t["vix_chg5"] = t.vix.pct_change(5) * 100
    return t.shift(1)        # only what was known at yesterday's close
