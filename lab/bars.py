"""Historical intraday bars for the strategy lab: Yahoo 5-minute (last 60 days) and 1-hour (last 2 years)
for every Nifty 500 / F&O stock and the main indices, cached on disk. Plain polite requests, one symbol
at a time; on HTTP 429 we wait, we do not work around it."""
import csv
import io
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sm import market as M          # noqa: E402

ROOT = M.DATA / "bars"
INDICES = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN", "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
           "MIDCPNIFTY": "NIFTY_MID_SELECT.NS", "NIFTYIT": "^CNXIT", "NIFTYAUTO": "^CNXAUTO", "NIFTYPHARMA": "^CNXPHARMA",
           "NIFTYMETAL": "^CNXMETAL", "NIFTYFMCG": "^CNXFMCG", "NIFTYREALTY": "^CNXREALTY", "NIFTYENERGY": "^CNXENERGY",
           "NIFTYPSUBANK": "^CNXPSUBANK", "NIFTYINFRA": "^CNXINFRA", "NIFTYMIDCAP": "NIFTY_MIDCAP_100.NS", "INDIAVIX": "^INDIAVIX"}
RANGE = {"5m": "60d", "60m": "730d", "1d": "5y"}


def yahoo(sym):
    return INDICES.get(sym, sym + ".NS")


def universe():
    """Nifty 500 plus every F&O stock (NSE's own lists)."""
    b = M.get("https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv").decode()
    syms = {r["Symbol"] for r in csv.DictReader(io.StringIO(b))}
    try:
        q = urllib.parse.quote("SECURITIES IN F&O")
        rows = M.get_json(f"https://www.nseindia.com/api/NextApi/apiClient/marketWatchApi?functionName=getIndicesData&symbol={q}")["data"]["data"]
        syms |= {r["symbol"] for r in rows if r.get("series") == "EQ"}
    except Exception:
        pass
    return sorted(syms)


def download(sym, tf):
    y = urllib.parse.quote(yahoo(sym))
    d = M.get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{y}?interval={tf}&range={RANGE[tf]}", 30)["chart"]["result"][0]
    o = d["indicators"]["quote"][0]
    df = pd.DataFrame({k: o[k] for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(d["timestamp"], unit="s", utc=True).tz_convert(M.IST)).dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)
    if tf != "1d":       # regular session only
        t = df.index.hour * 60 + df.index.minute
        df = df[(t >= 555) & (t < 930)]
    return df[~df.index.duplicated(keep="last")]


def path(sym, tf):
    return ROOT / tf / f"{sym}.pkl"


def save(sym, tf, df):
    """Merge with what is already on disk so the history keeps growing day after day."""
    p = path(sym, tf)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = pd.read_pickle(p)
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_pickle(p)


def load(sym, tf):
    p = path(sym, tf)
    return pd.read_pickle(p) if p.exists() else None


def fetch_all(syms, tfs=("5m", "60m", "1d"), fresh_hours=6, log=print):
    done = 0
    for i, s in enumerate(syms):
        for tf in tfs:
            p = path(s, tf)
            if p.exists() and time.time() - p.stat().st_mtime < fresh_hours * 3600:
                continue
            for attempt in range(4):
                try:
                    save(s, tf, download(s, tf))
                    done += 1
                    break
                except Exception as e:
                    if "429" in str(e):
                        log(f"429 from Yahoo, waiting 90s ({s} {tf})")
                        time.sleep(90)
                        continue
                    log(f"skip {s} {tf}: {str(e)[:80]}")
                    break
            time.sleep(0.4)
        if i % 25 == 0:
            log(f"{i}/{len(syms)} symbols, {done} files")
    log(f"done: {done} files")


if __name__ == "__main__":
    syms = list(INDICES) + universe()
    print(len(syms), "symbols", flush=True)
    fetch_all(syms, log=lambda m: print(m, flush=True))
