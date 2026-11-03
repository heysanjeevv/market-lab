"""Live state of the sector map.

- every NSE index (allIndices) every 15 s: today, week, month, year, PE, breadth
- every Nifty 500 stock (NSE market-watch, one call) every 15 s: price, change, volume, value
- the sector indices Yahoo carries, and the stocks of whatever sector or stock is open on the
  page, from Yahoo's batch feed every 5 s (2-3 s behind NSE)
- each sector index's members (NSE market-watch) once an hour
Every 5 minutes the sectors' moves are recorded, so the day can be replayed and each bubble
can show where it was 30 minutes ago."""
import json
import threading
import time
import urllib.parse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import market as M
from . import universe as U

SPARK = "https://query2.finance.yahoo.com/v7/finance/spark?symbols={}&range=1d&interval=1m"
DAYS = M.DATA / "days"
DAYS.mkdir(exist_ok=True)
S = {"indices": {}, "stocks": {}, "members": {}, "spark": {}, "focus": [], "focus_until": 0, "day": {"snaps": []},
     "time": None, "errors": 0, "slow_until": 0}
LOCK = threading.Lock()


def all_indices():
    d = M.get_json("https://www.nseindia.com/api/allIndices")
    out = {}
    for r in d["data"]:
        f = lambda k: U._f(r.get(k))
        last, wk = f("last"), f("oneWeekAgoVal")
        out[r["index"]] = {"name": r["index"], "last": last, "pct": f("percentChange"), "chg": f("variation"), "prev": f("previousClose"),
                           "w1": round((last / wk - 1) * 100, 2) if last and wk else None, "m1": f("perChange30d"), "y1": f("perChange365d"),
                           "pe": f("pe"), "adv": f("advances"), "dec": f("declines"), "high52": f("yearHigh"), "low52": f("yearLow"),
                           "open": f("open")}
    return out


def watch(index):
    q = urllib.parse.quote(index)
    d = M.get_json(f"https://www.nseindia.com/api/NextApi/apiClient/marketWatchApi?functionName=getIndicesData&symbol={q}")["data"]["data"]
    return {r["symbol"]: {"sym": r["symbol"], "ltp": r.get("lastPrice"), "pct": r.get("pChange"), "prev": r.get("previousClose"),
                          "open": r.get("open"), "high": r.get("dayHigh"), "low": r.get("dayLow"), "volume": r.get("totalTradedVolume") or 0,
                          "value_cr": round((r.get("totalTradedValue") or 0) / 1e7, 2), "ffmc": r.get("ffmc") or 0,
                          "high52": r.get("yearHigh"), "low52": r.get("yearLow"), "name": (r.get("companyName") or r["symbol"]).title()}
            for r in d if r.get("series") == "EQ"}


def spark(symbols):
    out = {}
    for i in range(0, len(symbols), 20):
        try:
            d = json.loads(M.get(SPARK.format(",".join(urllib.parse.quote(s) for s in symbols[i:i + 20]))))
        except Exception as e:
            S["errors"] += 1
            if "429" in str(e):
                S["slow_until"] = time.time() + 120
                break
            continue
        for res in d["spark"]["result"]:
            r = (res.get("response") or [None])[0]
            if not r or not r.get("timestamp"):
                continue
            m = r["meta"]
            cl = pd.Series(r["indicators"]["quote"][0]["close"], index=pd.to_datetime(r["timestamp"], unit="s", utc=True)).dropna()
            out[res["symbol"]] = {"ltp": m.get("regularMarketPrice"), "prev": m.get("chartPreviousClose"), "time": m.get("regularMarketTime"),
                                  "closes": cl}
        time.sleep(0.08)
    return out


def focus(syms):
    """The page asks for these stocks to be refreshed every few seconds (the open sector or stock)."""
    S["focus"] = [s for s in syms if s in U.U["list"]][:120]
    S["focus_until"] = time.time() + 60


def price(sym):
    """Best live price for a stock: Yahoo (seconds old) if it is being watched, else NSE (a minute)."""
    st = S["stocks"].get(sym) or {}
    y = S["spark"].get(sym + ".NS")
    if y and y["ltp"] and time.time() - (y.get("time") or 0) < 120:
        prev = st.get("prev") or y["prev"]
        return {**st, "ltp": y["ltp"], "pct": round((y["ltp"] / prev - 1) * 100, 2) if prev else st.get("pct"), "live": True}
    return st


def index_value(name):
    """An index's level and move, Yahoo-fresh when Yahoo carries it."""
    ix = dict(S["indices"].get(name) or {})
    y = U.YAHOO.get(name)
    g = S["spark"].get(y) if y else None
    if g and g["ltp"] and g["prev"]:
        ix.update(last=round(g["ltp"], 2), pct=round((g["ltp"] / g["prev"] - 1) * 100, 2), live=True)
    return ix


def record():
    """Every 5 minutes: each sector's move, for the replay and the 30-minute tails."""
    m = M.minute_now()
    if not (0 <= m <= 375):
        return
    d = S["day"]
    if d["snaps"] and d["snaps"][-1]["m"] > m - 5:
        return
    snap = {"m": m, "nifty": index_value("NIFTY 50").get("pct"), "sec": {}}
    for sec in U.SECTORS:
        v = index_value(sec["index"]).get("pct")
        if v is not None:
            snap["sec"][sec["name"]] = v
    d["snaps"].append(snap)
    (DAYS / f"{M.now_ist().date()}.json").write_text(json.dumps(d))


def run(log=print):
    last = {"idx": 0, "watch": 0, "members": 0, "fast": 0, "day": None}
    while True:
        try:
            today = M.now_ist().date()
            if last["day"] != today:
                p = DAYS / f"{today}.json"
                S["day"] = json.loads(p.read_text()) if p.exists() else {"snaps": []}
                last["day"] = today
            live = M.is_open()
            if time.time() - last["idx"] >= (15 if live else 300):
                S["indices"] = all_indices() or S["indices"]
                last["idx"] = time.time()
            if time.time() - last["watch"] >= (15 if live else 300):
                S["stocks"] = watch("NIFTY 500") or S["stocks"]
                last["watch"] = time.time()
            if time.time() - last["members"] >= 3600:
                mem = {}
                for sec in U.SECTORS:
                    try:
                        mem[sec["index"]] = watch(sec["index"])
                    except Exception:
                        pass
                    time.sleep(0.3)
                S["members"] = mem or S["members"]
                last["members"] = time.time()
            if time.time() > S["slow_until"] and (live or time.time() - last["fast"] > 300):
                ys = list(dict.fromkeys(U.YAHOO.values()))
                if time.time() < S["focus_until"]:
                    ys += [s + ".NS" for s in S["focus"]]
                got = spark(ys)
                with LOCK:
                    S["spark"].update(got)
                last["fast"] = time.time()
            record()
            S["time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        except Exception as e:
            log(f"live: {e}")
        time.sleep(5 if M.is_open() else 30)
