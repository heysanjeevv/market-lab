"""The stocks behind the map: the Nifty 500, each with NSE's four-level classification
(macro / sector / industry / basic industry), market value, PE, delivery and 52-week range
(NSE's public quote endpoint, refreshed weekly), and a year of daily bars (Yahoo, refreshed
daily) for week / month / 3-month moves, the 50-day average, normal volume and the sector
histories. Both are built slowly in the background, one polite request at a time."""
import csv
import io
import json
import time
import urllib.parse

import numpy as np
import pandas as pd

from . import market as M

CLS = M.DATA / "classification.json"
HIST = M.DATA / "history"
HIST.mkdir(exist_ok=True)

# the 20 sectors on the map: NSE index, group column, and which stocks belong (NSE classification)
SECTORS = [
    # NSE's market-watch does not list thematic index members, so Railways is the railway PSUs and wagon makers by name
    {"name": "Railways", "index": "NIFTY INDIA RAILWAYS PSU", "group": "Industrials & PSU",
     "symbols": ["IRCTC", "IRFC", "RVNL", "RAILTEL", "IRCON", "RITES", "CONCOR", "BEML", "TITAGARH", "JWL", "TEXRAIL", "RKFORGE", "HBLENGINE"]},
    {"name": "Energy", "index": "NIFTY ENERGY", "group": "Commodities & energy", "sector": ["Oil Gas & Consumable Fuels", "Power"]},
    {"name": "Infra", "index": "NIFTY INFRASTRUCTURE", "group": "Industrials & PSU", "sector": ["Construction"]},
    {"name": "Cons Durables", "index": "NIFTY CONSUMER DURABLES", "group": "Consumption", "sector": ["Consumer Durables"]},
    {"name": "Metal", "index": "NIFTY METAL", "group": "Commodities & energy", "sector": ["Metals & Mining"]},
    {"name": "Tourism", "index": "NIFTY INDIA TOURISM", "group": "Consumption", "basic": ["Hotels & Resorts", "Airline", "Tour, Travel Related Services", "Restaurants", "Airport & Airport services"]},
    {"name": "Defence", "index": "NIFTY INDIA DEFENCE", "group": "Industrials & PSU", "basic": ["Aerospace & Defense", "Ship Building & Allied Services", "Explosives"]},
    {"name": "Auto", "index": "NIFTY AUTO", "group": "Consumption", "sector": ["Automobile and Auto Components"]},
    {"name": "Chemicals", "index": "NIFTY CHEMICALS", "group": "Commodities & energy", "sector": ["Chemicals"]},
    {"name": "PSU Bank", "index": "NIFTY PSU BANK", "group": "Financials", "basic": ["Public Sector Bank"]},
    {"name": "Cap Markets", "index": "NIFTY CAPITAL MARKETS", "group": "Financials", "industry": ["Capital Markets"]},
    {"name": "Realty", "index": "NIFTY REALTY", "group": "Consumption", "sector": ["Realty"]},
    {"name": "FMCG", "index": "NIFTY FMCG", "group": "Consumption", "sector": ["Fast Moving Consumer Goods"]},
    {"name": "Fin Services", "index": "NIFTY FINANCIAL SERVICES", "group": "Financials", "sector": ["Financial Services"], "not_basic": ["Private Sector Bank", "Public Sector Bank"]},
    {"name": "Healthcare", "index": "NIFTY HEALTHCARE INDEX", "group": "Healthcare", "sector": ["Healthcare"], "not_basic": ["Pharmaceuticals"]},
    {"name": "IT", "index": "NIFTY IT", "group": "Tech & media", "sector": ["Information Technology"]},
    {"name": "Bank", "index": "NIFTY BANK", "group": "Financials", "industry": ["Banks"]},
    {"name": "Pharma", "index": "NIFTY PHARMA", "group": "Healthcare", "basic": ["Pharmaceuticals"]},
    {"name": "Media", "index": "NIFTY MEDIA", "group": "Tech & media", "sector": ["Media Entertainment & Publication"]},
    {"name": "Cement", "index": "NIFTY CEMENT", "group": "Commodities & energy", "sector": ["Construction Materials"]},
]
GROUPS = ["Tech & media", "Financials", "Healthcare", "Consumption", "Commodities & energy", "Industrials & PSU"]
YAHOO = {"NIFTY 50": "^NSEI", "NIFTY BANK": "^NSEBANK", "NIFTY IT": "^CNXIT", "NIFTY AUTO": "^CNXAUTO", "NIFTY PHARMA": "^CNXPHARMA",
         "NIFTY FMCG": "^CNXFMCG", "NIFTY METAL": "^CNXMETAL", "NIFTY REALTY": "^CNXREALTY", "NIFTY ENERGY": "^CNXENERGY",
         "NIFTY PSU BANK": "^CNXPSUBANK", "NIFTY MEDIA": "^CNXMEDIA", "NIFTY INFRASTRUCTURE": "^CNXINFRA",
         "NIFTY FINANCIAL SERVICES": "NIFTY_FIN_SERVICE.NS", "NIFTY HEALTHCARE INDEX": "NIFTY_HEALTHCARE.NS",
         "NIFTY CONSUMER DURABLES": "NIFTY_CONSR_DURBL.NS"}

U = {"list": {}, "cls": {}, "hist": {}, "progress": ""}


def nifty500():
    b = M.get("https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv").decode()
    return {r["Symbol"]: {"name": r["Company Name"], "industry": r["Industry"]} for r in csv.DictReader(io.StringIO(b))}


def quote(sym):
    d = M.get_json(f"https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi?functionName=getSymbolData&marketType=N&series=EQ&symbol={urllib.parse.quote(sym)}", 15)
    e = d["equityResponse"][0]
    si, ti, pi, md = e.get("secInfo") or {}, e.get("tradeInfo") or {}, e.get("priceInfo") or {}, e.get("metaData") or {}
    return {"macro": si.get("macro"), "sector": si.get("sector"), "industry": si.get("industryInfo") or si.get("industry"),
            "basic": si.get("basicIndustry"), "index": si.get("index"), "pe": _f(si.get("pdSymbolPe")), "sector_pe": _f(si.get("pdSectorPe")),
            "mcap": ti.get("totalMarketCap"), "ffmc": ti.get("ffmc"), "delivery": _f(ti.get("deliveryToTradedQuantity")),
            "volume": ti.get("totalTradedVolume"), "high52": pi.get("yearHigh"), "low52": pi.get("yearLow"),
            "name": md.get("companyName"), "ltp": md.get("lastPrice") or ti.get("lastPrice"), "pct": md.get("pChange"),
            "fetched": time.time()}


def _f(x):
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def build_classification(log=print):
    """NSE classification for every Nifty 500 stock; resumes where it stopped, weekly refresh."""
    cls = json.loads(CLS.read_text()) if CLS.exists() else {}
    todo = [s for s in U["list"] if s not in cls or time.time() - cls[s].get("fetched", 0) > 7 * 86400]
    for i, s in enumerate(todo):
        try:
            cls[s] = quote(s)
        except Exception:
            time.sleep(2)
            continue
        if i % 25 == 0:
            U["progress"] = f"classifying stocks {len(cls)}/{len(U['list'])}"
            CLS.write_text(json.dumps(cls))
            U["cls"] = dict(cls)
        time.sleep(0.25)
    CLS.write_text(json.dumps(cls))
    U["cls"] = cls
    log(f"classification: {len(cls)} stocks")


def yahoo_daily(y, rng="1y"):
    d = M.get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(y)}?interval=1d&range={rng}")["chart"]["result"][0]
    o = d["indicators"]["quote"][0]
    return pd.DataFrame({"close": o["close"], "high": o["high"], "volume": o["volume"]},
                        index=pd.to_datetime(d["timestamp"], unit="s", utc=True).tz_convert(M.IST).normalize().tz_localize(None)).dropna(subset=["close"])


def build_history(log=print):
    """A year of daily bars per stock, stored per day so a restart does not fetch again."""
    day = M.now_ist().date().isoformat()
    p = HIST / f"{day}.pkl"
    if p.exists():
        U["hist"] = pd.read_pickle(p)
        return
    old = sorted(HIST.glob("*.pkl"))
    hist = pd.read_pickle(old[-1]) if old else {}
    U["hist"] = hist
    for i, s in enumerate(U["list"]):
        try:
            hist[s] = yahoo_daily(s + ".NS")
        except Exception:
            pass
        if i % 25 == 0:
            U["progress"] = f"loading price history {i}/{len(U['list'])}"
        time.sleep(0.2)
    for name, y in {**YAHOO}.items():
        try:
            hist["IDX:" + name] = yahoo_daily(y)
        except Exception:
            pass
    pd.to_pickle(hist, p)
    for q in old:
        q.unlink(missing_ok=True)
    U["hist"] = hist
    U["progress"] = ""
    log(f"history: {len(hist)} series")


def members(sec, index_members=()):
    """Nifty 500 stocks NSE classifies into this sector, plus the sector index's own members."""
    out = [s for s in index_members if s in U["list"]]
    if "symbols" in sec:
        return [s for s in sec["symbols"] if s in U["list"] or s in U["cls"]]
    for s, c in U["cls"].items():
        if s not in U["list"]:
            continue
        ok = False
        if "sector" in sec and c.get("sector") in sec["sector"]:
            ok = True
        if "industry" in sec and c.get("industry") in sec["industry"]:
            ok = True
        if "basic" in sec and c.get("basic") in sec["basic"]:
            ok = True
        if c.get("basic") in sec.get("not_basic", []):
            ok = False
        if ok and s not in out:
            out.append(s)
    return out


def stats(sym):
    """Week / month / 3-month moves, 50-day average and normal volume from the daily bars."""
    h = U["hist"].get(sym)
    if h is None or len(h) < 30:
        return {}
    c = h["close"]
    r = lambda n: float(c.iloc[-1] / c.iloc[-1 - n] - 1) * 100 if len(c) > n else None
    return {"w1": r(5), "m1": r(21), "m3": r(63), "dma50": float(c.iloc[-50:].mean()), "vol60": float(h["volume"].iloc[-60:].mean() or 0),
            "high52": float(h["high"].iloc[-250:].max()), "last_close": float(c.iloc[-1])}


def sector_series(syms, days=70):
    """An equal-weight index of the sector's stocks (each rebased to 100), median of their paths."""
    cols = [U["hist"][s]["close"].iloc[-days:] for s in syms if s in U["hist"] and len(U["hist"][s]) > days]
    if not cols:
        return None
    df = pd.concat(cols, axis=1).ffill().dropna()
    return (df / df.iloc[0] * 100).median(axis=1)
