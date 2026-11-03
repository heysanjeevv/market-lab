"""Live tape for Smart Money: every F&O stock and Nifty, Bank Nifty, Sensex.

Prices: Yahoo's batch "spark" endpoint (20 symbols a call, 1-minute closes since the open and
the last price, 2-3 seconds behind NSE): the three indices every 4 seconds, every stock every
12 seconds. NSE's market-watch adds the official open, high, low, value traded and free-float
size once a minute; daily bars per stock (once a day) give the previous day's levels and a
normal day's range. Structure, order blocks and gaps are read from 5-minute bars built out of
the 1-minute closes. A broker WebSocket with your own key would give true ticks."""
import csv
import io
import json
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import market as M
from . import smc

SPARK = "https://query2.finance.yahoo.com/v7/finance/spark?symbols={}&range=1d&interval=1m"
INDICES = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
ORIGIN = pd.Timestamp("2000-01-03 03:45", tz="UTC")

# trading groups for the sector chips: NSE's industry, split where one industry hides several stories
INSURANCE = {"SBILIFE", "HDFCLIFE", "ICICIPRULI", "ICICIGI", "LICI", "POLICYBZR", "NIACL", "GICRE", "STARHEALTH", "MAXHEALTH_X"}
CAP_MKTS = {"BSE", "MCX", "CDSL", "CAMS", "ANGELONE", "KFINTECH", "NUVAMA", "360ONE", "HDFCAMC", "NAM-INDIA", "MOTILALOFS", "IIFL", "UTIAMC", "ABSLAMC"}
PSU_BANK = {"SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "INDIANB", "BANKINDIA", "IOB", "UCOBANK", "CENTRALBK", "MAHABANK"}
DEFENCE = {"HAL", "BEL", "BDL", "MAZDOCK", "COCHINSHIP", "GRSE", "BEML", "DATAPATTNS", "SOLARINDS", "ASTRAMICRO", "PARAS", "MIDHANI"}
NEW_AGE = {"ETERNAL", "ZOMATO", "PAYTM", "NYKAA", "SWIGGY", "DELHIVERY_X", "IXIGO", "CARTRADE", "NAUKRI"}
LOGISTICS = {"DELHIVERY", "CONCOR", "BLUEDART", "TCI", "ADANIPORTS", "GMRAIRPORT", "JSWINFRA", "INDIGO"}
TELECOM = {"BHARTIARTL", "IDEA", "INDUSTOWER", "TATACOMM", "HFCL", "TEJASNET"}
GROUP = {"Oil Gas & Consumable Fuels": "OIL & GAS", "Consumer Durables": "CONSUMER", "Consumer Services": "CONSUMER", "Textiles": "CONSUMER",
         "Capital Goods": "CAP GOODS", "Construction": "CAP GOODS", "Metals & Mining": "METAL", "Construction Materials": "CEMENT",
         "Power": "POWER", "Automobile and Auto Components": "AUTO", "Healthcare": "HEALTHCARE", "Chemicals": "CHEM", "Realty": "REALTY",
         "Information Technology": "IT", "Fast Moving Consumer Goods": "FMCG", "Telecommunication": "TELECOM", "Services": "LOGISTICS",
         "Media Entertainment & Publication": "OTHERS", "Diversified": "OTHERS", "Forest Materials": "OTHERS"}
PHARMA_HINT = {"SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "AUROPHARMA", "ZYDUSLIFE", "ALKEM", "TORNTPHARM", "BIOCON",
               "GLENMARK", "LAURUSLABS", "MANKIND", "IPCALAB", "GRANULES", "PPLPHARMA", "AJANTPHARM", "NATCOPHARM", "ABBOTINDIA"}


def sector_of(sym, industry):
    for s, g in ((INSURANCE, "INSURANCE"), (CAP_MKTS, "CAP MKTS"), (PSU_BANK, "PSU BANK"), (DEFENCE, "DEFENCE"),
                 (NEW_AGE, "NEW-AGE"), (LOGISTICS, "LOGISTICS"), (TELECOM, "TELECOM")):
        if sym in s:
            return g
    if industry == "Financial Services":
        return "BANK" if sym.endswith("BANK") or sym in {"HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "FEDERALBNK",
                                                         "IDFCFIRSTB", "BANDHANBNK", "AUBANK", "RBLBANK", "YESBANK", "CUB"} else "NBFC"
    if industry == "Healthcare":
        return "PHARMA" if sym in PHARMA_HINT else "HEALTHCARE"
    return GROUP.get(industry, "OTHERS")


STATE = {"stocks": {}, "indices": {}, "meta": {}, "daily": {}, "daily_idx": {}, "industry": {}, "trail": {},
         "time": None, "errors": 0, "slow_until": 0}
GOT, LOCK = {}, threading.Lock()


def fno_list():
    q = urllib.parse.quote("SECURITIES IN F&O")
    rows = M.get_json(f"https://www.nseindia.com/api/NextApi/apiClient/marketWatchApi?functionName=getIndicesData&symbol={q}")["data"]["data"]
    return {r["symbol"]: {"name": (r.get("companyName") or r["symbol"]).title(), "open": r.get("open"), "high": r.get("dayHigh"),
                          "low": r.get("dayLow"), "prev": r.get("previousClose"), "ltp": r.get("lastPrice"),
                          "value_cr": round((r.get("totalTradedValue") or 0) / 1e7, 1), "ffmc": r.get("ffmc") or 0}
            for r in rows if r.get("series") == "EQ"}


def industries():
    b = M.get("https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv").decode()
    return {r["Symbol"]: r["Industry"] for r in csv.DictReader(io.StringIO(b))}


def spark(symbols):
    out = {}
    d = json.loads(M.get(SPARK.format(",".join(urllib.parse.quote(s) for s in symbols))))
    for res in d["spark"]["result"]:
        r = (res.get("response") or [None])[0]
        if not r or not r.get("timestamp"):
            continue
        s = pd.Series(r["indicators"]["quote"][0]["close"], index=pd.to_datetime(r["timestamp"], unit="s", utc=True)).dropna()
        m = r["meta"]
        out[res["symbol"]] = {"closes": s, "ltp": m.get("regularMarketPrice"), "prev": m.get("chartPreviousClose"),
                              "time": m.get("regularMarketTime")}
    return out


def fetch(ysyms):
    for i in range(0, len(ysyms), 20):
        try:
            GOT.update(spark(ysyms[i:i + 20]))
            STATE["errors"] = 0
        except Exception as e:
            STATE["errors"] += 1
            if "429" in str(e):
                STATE["slow_until"] = time.time() + 120      # Yahoo asks us to slow down: do
                return
        time.sleep(0.08)


def daily_bars(y, rng="3mo"):
    d = M.get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(y)}?interval=1d&range={rng}")["chart"]["result"][0]
    o = d["indicators"]["quote"][0]
    return pd.DataFrame({k: o[k] for k in ("open", "high", "low", "close", "volume")},
                        index=pd.to_datetime(d["timestamp"], unit="s", utc=True)).dropna(subset=["close"])


def load_daily(symbols, log=print):
    out, today = {}, M.now_ist().date()
    for i, s in enumerate(symbols):
        try:
            df = daily_bars(s + ".NS")
            past = df[[t.tz_convert(M.IST).date() < today for t in df.index]]
            if len(past) >= 15:
                rng = (past["high"] - past["low"]) / past["close"] * 100
                out[s] = {"pdh": float(past["high"].iloc[-1]), "pdl": float(past["low"].iloc[-1]),
                          "pdc": float(past["close"].iloc[-1]), "range": float(rng.iloc[-20:].mean())}
        except Exception:
            pass
        if i % 50 == 0:
            log(f"daily levels {i}/{len(symbols)}")
        time.sleep(0.2)
    return out


def bars5(closes, open_px=None):
    if closes is None or len(closes) == 0:
        return None
    df = closes.resample("5min", origin=ORIGIN).agg(["first", "max", "min", "last"]).dropna()
    df.columns = ["open", "high", "low", "close"]
    if open_px and len(df):
        df.iloc[0, 0] = open_px
        df.iloc[0, 1] = max(df.iloc[0, 1], open_px)
        df.iloc[0, 2] = min(df.iloc[0, 2], open_px)
    df["volume"] = 0.0
    return df


def path(closes, base):
    if closes is None or not base:
        return []
    m, _ = M.session_minutes(closes.index)
    return [[int(a), round(float(b), 3)] for a, b in zip(m, (closes.values / base - 1) * 100) if 0 <= a <= 375]


def _read(df, daily=None):
    try:
        r = smc.read(df, 2, daily) if df is not None and len(df) >= 12 else None
    except Exception:
        return None
    if not r:
        return None
    st = r["structure"]["last"]
    return {"struct": st and {"type": st["type"], "dir": st["dir"], "time": st["time"]},
            "ob": (r["order_blocks"] or [None])[0], "fvg": (r["fvgs"] or [None])[0], "liq": r["liquidity"],
            "ev": [[int(M.session_minutes(pd.DatetimeIndex([pd.Timestamp(e["time"], unit="s", tz="UTC")]))[0][0]), 1 if e["dir"] == "bull" else -1]
                   for e in r["chart"]["events"] if e["time"] >= int(df.index[0].timestamp())]}


def cycle(stocks_too, structure):
    meta = STATE["meta"]
    syms = list(meta)
    fetch(([s + ".NS" for s in syms] if stocks_too else []) + list(INDICES.values()))
    now_m = M.minute_now()
    if stocks_too:
        stocks = {}
        for s in syms:
            g, nse, dly = GOT.get(s + ".NS"), meta[s], STATE["daily"].get(s)
            ltp = (g or {}).get("ltp") or nse.get("ltp")
            prev = nse.get("prev") or (g or {}).get("prev")
            cl = g["closes"] if g else None
            opn = nse.get("open") or (float(cl.iloc[0]) if cl is not None and len(cl) else None)
            if not (ltp and prev and opn):
                continue
            fo = (ltp / opn - 1) * 100
            ind = STATE["industry"].get(s, "")
            row = {"sym": s, "name": nse["name"], "sector": sector_of(s, ind), "ltp": round(ltp, 2), "pct": round((ltp / prev - 1) * 100, 2),
                   "from_open": round(fo, 2), "disp": round(fo / ((dly or {}).get("range") or 2.0), 2), "path": path(cl, opn),
                   "value_cr": nse.get("value_cr"), "ffmc": nse.get("ffmc", 0)}
            b = bars5(cl, opn)
            if b is not None:
                row["bars"] = [[round(float(x), 2) for x in r] for r in b[["open", "high", "low", "close"]].values]
            if dly:
                row["pdh"], row["pdl"] = round(dly["pdh"], 2), round(dly["pdl"], 2)
                row["liq"] = [k for k, hit in (("PDH", (nse.get("high") or 0) > dly["pdh"]), ("PDL", 0 < (nse.get("low") or 1e12) < dly["pdl"])) if hit]
            old = STATE["stocks"].get(s, {})
            if structure or "read" not in old:
                rd = _read(b)
                row["read"] = rd and {k: rd[k] for k in ("struct", "ob", "fvg", "ev")}
            else:
                row["read"] = old.get("read")
            # where the stock stood 15 minutes ago, for the bead's trail
            p = row["path"]
            row["trail"] = next((v for m, v in reversed(p) if m <= now_m - 15), None) if p else None
            stocks[s] = row
        bull = sorted([r for r in stocks.values() if r["from_open"] > 0], key=lambda r: -r["disp"])
        bear = sorted([r for r in stocks.values() if r["from_open"] <= 0], key=lambda r: r["disp"])
        for i, r in enumerate(bull):
            r["rank"], r["side"] = i + 1, "bull"
        for i, r in enumerate(bear):
            r["rank"], r["side"] = i + 1, "bear"
    else:
        stocks = STATE["stocks"]
    idx = {}
    for k, y in INDICES.items():
        g = GOT.get(y)
        if not g or not g["ltp"]:
            continue
        cl = g["closes"]
        base = float(cl.iloc[0]) if len(cl) else g["ltp"]
        b = bars5(cl, base)
        old = STATE["indices"].get(k, {})
        rd = _read(b, STATE["daily_idx"].get(k)) if structure or not old.get("read") else old.get("read")
        idx[k] = {"ltp": round(g["ltp"], 2), "pct": round((g["ltp"] / g["prev"] - 1) * 100, 2) if g["prev"] else None,
                  "from_open": round((g["ltp"] / base - 1) * 100, 2), "path": path(cl, base),
                  "bars": [[round(float(x), 2) for x in r] for r in b[["open", "high", "low", "close"]].values] if b is not None else [],
                  "read": rd, "expiry": next_expiry(k), "delay": int(time.time() - g["time"]) if g.get("time") else None}
    with LOCK:
        STATE.update(stocks=stocks, indices=idx, time=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def next_expiry(key):
    """Nifty weekly Tuesday, Bank Nifty last Tuesday of the month, Sensex weekly Thursday."""
    hol = M.holidays()
    d = M.now_ist().date()
    for k in range(45):
        x = d + timedelta(days=k)
        if x.weekday() != (3 if key == "SENSEX" else 1):
            continue
        if key == "BANKNIFTY" and (x + timedelta(days=7)).month == x.month:
            continue
        e = x
        while not M.is_trading_day(e, hol):
            e -= timedelta(days=1)
        if e >= d:
            return e.isoformat()
    return None


def snapshot(paths=True):
    with LOCK:
        stocks = list(STATE["stocks"].values())
        idx = dict(STATE["indices"])
    if not paths:
        stocks = [{k: v for k, v in r.items() if k != "path"} for r in stocks]
    secs = {}
    for r in stocks:
        secs.setdefault(r["sector"], []).append(r)
    sectors = sorted([{"name": k, "pct": round(float(np.mean([r["pct"] for r in v])), 2), "up": sum(r["pct"] > 0 for r in v),
                       "down": sum(r["pct"] < 0 for r in v), "n": len(v)} for k, v in secs.items()], key=lambda s: -s["pct"])
    return {"time": STATE["time"], "market": M.market_status(), "errors": STATE["errors"], "indices": idx, "stocks": stocks,
            "sectors": sectors, "bull": [r["sym"] for r in sorted(stocks, key=lambda r: r.get("rank", 999)) if r.get("side") == "bull"],
            "bear": [r["sym"] for r in sorted(stocks, key=lambda r: r.get("rank", 999)) if r.get("side") == "bear"]}


def quote(sym):
    """Fresh last price of one symbol (for the pop-up chart's live candle)."""
    y = INDICES.get(sym, sym if sym.startswith("^") else sym + ".NS")
    m = M.get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(y)}?interval=1m&range=1d", 8)["chart"]["result"][0]["meta"]
    return {"ltp": m.get("regularMarketPrice"), "time": m.get("regularMarketTime"), "prev": m.get("chartPreviousClose")}


def chart(sym, tf="5m"):
    rng = {"1m": "5d", "5m": "1mo", "15m": "60d", "60m": "2y", "1d": "5y"}[tf]
    y = INDICES.get(sym, sym if sym.startswith("^") else sym + ".NS")
    d = M.get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(y)}?interval={tf}&range={rng}")["chart"]["result"][0]
    o = d["indicators"]["quote"][0]
    df = pd.DataFrame({k: o[k] for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(d["timestamp"], unit="s", utc=True)).dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)
    daily = None
    if tf != "1d":
        try:
            daily = daily_bars(y)
        except Exception:
            pass
    rd = smc.read(df, 2 if tf in ("1m", "5m") else 3, daily) if len(df) >= 30 else None
    m = d["meta"]
    return {"sym": sym, "tf": tf, "ltp": m.get("regularMarketPrice"), "prev": m.get("chartPreviousClose"),
            "candles": [{"time": int(t.timestamp()), "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
                        for t, r in df.iterrows()], "read": rd}


def run(log=print):
    last = {"day": None, "nse": 0, "stocks": 0, "struct": 0}
    while True:
        try:
            today = M.now_ist().date()
            if last["day"] != today:
                STATE["industry"], STATE["meta"] = industries(), fno_list()
                last.update(day=today, nse=time.time())
                log(f"{len(STATE['meta'])} F&O stocks; loading daily levels")
                STATE["daily_idx"] = {}
                for k, y in INDICES.items():
                    try:
                        STATE["daily_idx"][k] = daily_bars(y)
                    except Exception:
                        pass
                cycle(True, True)                 # show prices before the slow daily levels arrive
                STATE["daily"] = load_daily(list(STATE["meta"]), log)
            live = M.is_open()
            if time.time() - last["nse"] >= (60 if live else 1800):
                try:
                    STATE["meta"] = fno_list() or STATE["meta"]
                except Exception as e:
                    log(f"NSE list: {e}")
                last["nse"] = time.time()
            slow = time.time() < STATE["slow_until"]
            every = 12 if live and not slow else 60 if live else 300
            if time.time() - last["stocks"] >= every:
                st = time.time() - last["struct"] >= 60
                cycle(True, st)
                last["stocks"] = time.time()
                if st:
                    last["struct"] = time.time()
            elif live and not slow:
                cycle(False, False)
        except Exception as e:
            log(f"feed: {e}")
        time.sleep(4 if M.is_open() else 60)
