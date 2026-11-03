"""Market dashboard: Smart Money (every F&O stock and the three indices, live) on top and the
Sector Map (sectors, the stocks inside, one stock up close) below, on one page.
Run ./run.sh, then open http://localhost:8810.

Two data engines run side by side: sm/ (the Smart Money tape) and sx/ (the Sector Map)."""
import json
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from sm import feed as F
from sx import live as L
from sx import market as M
from sx import news as N
from sx import smc
from sx import universe as U
from lab import live as LV
from lab import swing_live as SW
from lab import research as RS
from lab.strategies import STRATEGIES

ROOT = Path(__file__).resolve().parent
PORT = 8810
_C = {}


def cached(key, secs, fn):
    if key in _C and time.time() - _C[key][0] < secs:
        return _C[key][1]
    v = fn()
    _C[key] = (time.time(), v)
    return v


def r2(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


# ---------- sectors ----------

def sec_members(sec):
    return U.members(sec, list((L.S["members"].get(sec["index"]) or {}).keys()))


def series_for(sec):
    """3-4 months of the sector: its NSE index from Yahoo when Yahoo has it, else the median of its stocks."""
    h = U.U["hist"].get("IDX:" + sec["index"])
    if h is not None and len(h) > 70:
        return h["close"].iloc[-70:] / h["close"].iloc[-70] * 100
    return U.sector_series(sec_members(sec))


def all_series():
    return cached(("series", M.now_ist().date().isoformat(), len(U.U["hist"])), 3600,
                  lambda: {s["name"]: series_for(s) for s in U.SECTORS})


def nifty_hist():
    h = U.U["hist"].get("IDX:NIFTY 50")
    return h["close"] if h is not None else None


def ret(series, n):
    if series is None or len(series) <= n:
        return None
    return float(series.iloc[-1] / series.iloc[-1 - n] - 1) * 100


def sector_move(sec, rng):
    ix = L.index_value(sec["index"])
    if rng == "today":
        return ix.get("pct")
    if rng in ("w1", "m1") and ix.get(rng) is not None:
        return ix.get(rng)
    s = all_series().get(sec["name"])
    return ret(s, {"w1": 5, "m1": 21, "m3": 63}[rng])


def nifty_move(rng):
    ix = L.index_value("NIFTY 50")
    if rng == "today":
        return ix.get("pct")
    if rng in ("w1", "m1") and ix.get(rng) is not None:
        return ix[rng]
    return ret(nifty_hist(), {"w1": 5, "m1": 21, "m3": 63}[rng])


def ranks_now(rng):
    v = {s["name"]: sector_move(s, rng) for s in U.SECTORS}
    order = sorted([k for k in v if v[k] is not None], key=lambda k: -v[k])
    return {k: i + 1 for i, k in enumerate(order)}, v


def rank_history(name, weeks=10):
    """Where the sector ranked among the 20 by its trailing month, at the end of each of the last weeks."""
    ser = all_series()
    out = []
    for w in range(weeks, -1, -1):
        end = -1 - w * 5
        vals = {}
        for k, s in ser.items():
            if s is not None and len(s) > 22 - end:
                vals[k] = float(s.iloc[end] / s.iloc[end - 21] - 1)
        order = sorted(vals, key=lambda k: -vals[k])
        if name in order:
            out.append({"date": str(ser[name].index[end].date()), "rank": order.index(name) + 1})
    return out


def themes():
    """Basic industries across the Nifty 500 with at least 5 stocks: their middle stock today and how many beat Nifty."""
    nifty = (L.index_value("NIFTY 50") or {}).get("pct") or 0
    by = {}
    for s, c in U.U["cls"].items():
        p = L.price(s)
        if c.get("basic") and p.get("pct") is not None:
            by.setdefault(c["basic"], []).append((s, p["pct"], c.get("sector")))
    out = []
    for b, rows in by.items():
        if len(rows) < 5:
            continue
        mid = float(np.median([r[1] for r in rows]))
        out.append({"name": b, "parent": rows[0][2], "n": len(rows), "beat": sum(r[1] > nifty for r in rows), "pct": round(mid, 2)})
    return sorted([t for t in out if t["pct"] > nifty], key=lambda t: -t["pct"])[:8]


def since(sec_name, iso):
    """How far the sector moved from the day of a headline to now."""
    s = all_series().get(sec_name)
    if s is None:
        return None
    d = pd.Timestamp(iso).tz_convert(M.IST).tz_localize(None).normalize()
    before = s[s.index < d]
    if not len(before):
        return None
    today = L.index_value(next(x["index"] for x in U.SECTORS if x["name"] == sec_name)).get("pct") or 0
    return round((float(s.iloc[-1] / before.iloc[-1]) * (1 + today / 100) - 1) * 100, 2)


def pins(items, dates):
    """Headlines bucketed onto the chart's days: per day a stack of N / R / P."""
    out = {}
    ds = [pd.Timestamp(d) for d in dates]
    for h in items:
        t = pd.Timestamp(h["time"]).tz_convert(M.IST).tz_localize(None).normalize()
        cand = [d for d in ds if d <= t]
        if not cand:
            continue
        k = str(cand[-1].date())
        out.setdefault(k, []).append({"tag": h["tag"][0], "title": h["title"], "source": h["source"], "link": h["link"]})
    return out


def rel_chart(a, b, items):
    """Two series rebased to 100 over the same days, with headline pins."""
    if a is None or b is None:
        return None
    df = pd.concat([a, b], axis=1).ffill().dropna().iloc[-55:]
    if len(df) < 5:
        return None
    df = df / df.iloc[0] * 100
    dates = [str(d.date()) for d in df.index]
    return {"dates": dates, "a": [r2(x) for x in df.iloc[:, 0]], "b": [r2(x) for x in df.iloc[:, 1]], "pins": pins(items, dates)}


def horizon(rng):
    rk, vals = ranks_now(rng)
    first = L.S["day"]["snaps"][0]["sec"] if L.S["day"]["snaps"] else {}
    rk_open = {k: i + 1 for i, k in enumerate(sorted(first, key=lambda k: -first[k]))} if first else {}
    snaps = L.S["day"]["snaps"]
    secs = []
    for sec in U.SECTORS:
        mem = sec_members(sec)
        size = sum((L.S["stocks"].get(s) or {}).get("ffmc", 0) for s in mem)
        ix = L.index_value(sec["index"])
        v = vals.get(sec["name"])
        trail = None
        if rng == "today" and len(snaps) > 6:
            trail = snaps[-7]["sec"].get(sec["name"])
        secs.append({"name": sec["name"], "index": sec["index"], "group": sec["group"], "v": r2(v), "size": size, "n": len(mem),
                     "rank": rk.get(sec["name"]), "rank_open": rk_open.get(sec["name"]), "trail": r2(trail), "live": bool(ix.get("live")),
                     "path": [[s["m"], s["sec"].get(sec["name"])] for s in snaps] if rng == "today" else []})
    climb = [{"name": s["name"], "from": s["rank_open"], "to": s["rank"]} for s in secs if s["rank_open"] and s["rank"]]
    if rng != "today":
        prev = {}
        for sec in U.SECTORS:
            sr = all_series().get(sec["name"])
            n = {"w1": 5, "m1": 21, "m3": 63}[rng]
            if sr is not None and len(sr) > n + 6:
                prev[sec["name"]] = float(sr.iloc[-6] / sr.iloc[-6 - n] - 1)
        order = sorted(prev, key=lambda k: -prev[k])
        climb = [{"name": k, "from": order.index(k) + 1, "to": rk[k]} for k in order if k in rk]
    news = []
    for sec in U.SECTORS:
        for h in N.sector(sec["name"], wait=False)[:4]:
            news.append({**h, "sector": sec["name"]})
    news.sort(key=lambda h: h["time"], reverse=True)
    for h in news[:14]:
        h["since"] = since(h["sector"], h["time"])
    mk = _C.get("mk", (0, []))[1]
    n = nifty_hist()
    chart = None
    if n is not None:
        c = n.iloc[-60:]
        dates = [str(d.date()) for d in c.index]
        chart = {"dates": dates, "a": [r2(x) for x in c], "pins": pins(mk + news, dates)}
    return {"range": rng, "nifty": {"v": r2(nifty_move(rng)), "last": L.index_value("NIFTY 50").get("last")}, "sectors": secs, "groups": U.GROUPS,
            "climb": climb, "themes": themes() if rng == "today" else [], "news": news[:14], "chart": chart,
            "snaps": [{"m": s["m"], "nifty": s["nifty"]} for s in snaps]}


def stock_row(s, c=None):
    c = c or U.U["cls"].get(s, {})
    p = L.price(s)
    st = U.stats(s)
    ltp = p.get("ltp")
    vr = (p.get("volume") or 0) / st["vol60"] if st.get("vol60") else None
    return {"sym": s, "name": (c.get("name") or p.get("name") or s), "group": c.get("industry") or "Other", "basic": c.get("basic"),
            "ltp": r2(ltp), "pct": r2(p.get("pct")), "mcap": c.get("mcap") or p.get("ffmc") or 0, "live": bool(p.get("live")),
            "w1": r2(st.get("w1")), "m1": r2(st.get("m1")), "m3": r2(st.get("m3")),
            "above50": (ltp > st["dma50"]) if ltp and st.get("dma50") else None, "near_high": bool(ltp and st.get("high52") and ltp >= st["high52"] * .95),
            "vol_ratio": r2(vr), "high52": r2(c.get("high52") or st.get("high52")), "dma50": r2(st.get("dma50"))}


def sector_view(name, rng):
    sec = next((x for x in U.SECTORS if x["name"].lower() == name.lower()), None)
    if not sec:
        return {"error": "unknown sector"}
    mem = sec_members(sec)
    L.focus(mem)
    rows = [stock_row(s) for s in mem]
    rows = [r for r in rows if r["ltp"]]
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    glist = [{"name": g, "n": len(v), "mid": r2(np.median([r["pct"] for r in v if r["pct"] is not None])), "above50": sum(r["above50"] for r in v),
              "stocks": sorted(v, key=lambda r: -(r["mcap"] or 0))} for g, v in groups.items()]
    glist.sort(key=lambda g: -sum(r["mcap"] or 0 for r in g["stocks"]))
    if len(glist) > 6:     # small groups fold into one
        rest = [r for g in glist[5:] for r in g["stocks"]]
        glist = glist[:5] + [{"name": "Other index members", "n": len(rest), "mid": r2(np.median([r["pct"] for r in rest])),
                              "above50": sum(r["above50"] for r in rest), "stocks": rest}]
    ix = L.index_value(sec["index"])
    rk, vals = ranks_now("today")
    snaps = L.S["day"]["snaps"]
    first = snaps[0]["sec"] if snaps else {}
    rk_open = sorted(first, key=lambda k: -first[k]).index(sec["name"]) + 1 if sec["name"] in first else None
    # stocks the sector index holds that NSE's classification files under another sector
    ix_mem = set((L.S["members"].get(sec["index"]) or {}).keys())
    own = set(U.members(sec))
    others = []
    for s in sorted(ix_mem - own):
        o = next((x for x in U.SECTORS if x is not sec and s in U.members(x)), None)
        if o:
            p = L.price(s)
            others.append({"sym": s, "sector": o["name"], "ltp": r2(p.get("ltp")), "pct": r2(p.get("pct"))})
    ser = all_series().get(sec["name"])
    wk_above = 0
    for r in rows:
        h = U.U["hist"].get(r["sym"])
        if h is not None and len(h) > 56:
            wk_above += int(h["close"].iloc[-6] > h["close"].iloc[-56:-6].mean())
    news = N.sector(sec["name"])
    for h in news:
        h["since"] = since(sec["name"], h["time"])
    mid = r2(np.median([r["pct"] for r in rows if r["pct"] is not None])) if rows else None
    nifty_pe = (L.S["indices"].get("NIFTY 50") or {}).get("pe")
    return {"name": sec["name"], "index": sec["index"], "group": sec["group"],
            "ix": {"last": ix.get("last"), "pct": ix.get("pct"), "w1": r2(sector_move(sec, "w1")), "m1": r2(sector_move(sec, "m1")),
                   "m3": r2(sector_move(sec, "m3")), "pe": ix.get("pe"), "live": bool(ix.get("live"))},
            "nifty": {"pct": r2(nifty_move("today")), "w1": r2(nifty_move("w1")), "m1": r2(nifty_move("m1")), "m3": r2(nifty_move("m3")), "pe": nifty_pe},
            "rank": rk.get(sec["name"]), "rank_open": rk_open, "n": len(rows), "middle": mid, "groups": glist, "others": others[:14],
            "rank_hist": rank_history(sec["name"]),
            "inside": {"above50": sum(r["above50"] for r in rows), "above50_wk": wk_above, "rose_week": sum((r["w1"] or 0) > 0 for r in rows),
                       "near_high": sum(r["near_high"] for r in rows), "busy": sum((r["vol_ratio"] or 0) > 1.5 for r in rows), "n": len(rows)},
            "movers": sorted([r for r in rows if r["w1"] is not None], key=lambda r: -r["w1"])[:6],
            "laggards": sorted([r for r in rows if r["w1"] is not None], key=lambda r: r["w1"])[:4],
            "news": news[:10], "chart": rel_chart(ser, nifty_hist(), news)}


def stock_view(sym):
    sym = sym.upper()
    c = U.U["cls"].get(sym) or {}
    try:
        q = cached(("q", sym), 30, lambda: U.quote(sym))
    except Exception:
        q = {}
    c = {**c, **{k: v for k, v in q.items() if v is not None and k in ("delivery", "volume", "pe", "high52", "low52", "mcap", "name", "basic", "industry", "sector")}}
    row = stock_row(sym, c)
    sec = next((x for x in U.SECTORS if sym in sec_members(x)), None)
    peers = []
    if sec:
        peers = [stock_row(s) for s in sec_members(sec)]
        peers = [p for p in peers if p["group"] == row["group"] and p["ltp"]]
    L.focus([sym] + [p["sym"] for p in peers])
    st = U.stats(sym)
    h = U.U["hist"].get(sym)
    sser = all_series().get(sec["name"]) if sec else None
    against = []
    if sec:
        sx = L.index_value(sec["index"])
        for k, lab in (("today", "Today"), ("w1", "Week"), ("m1", "1 month"), ("m3", "3 months")):
            a = row["pct"] if k == "today" else row[k]
            b = sx.get("pct") if k == "today" else sector_move(sec, k)
            against.append({"label": lab, "stock": r2(a), "sector": r2(b), "gap": r2(a - b) if a is not None and b is not None else None})
    rank = None
    if sec and row["w1"] is not None:
        ws = sorted([p for p in (stock_row(s) for s in sec_members(sec)) if p["w1"] is not None], key=lambda p: -p["w1"])
        rank = {"pos": next((i + 1 for i, p in enumerate(ws) if p["sym"] == sym), None), "of": len(ws)}
    lines = None
    if h is not None and len(h) > 63:
        a = h["close"].iloc[-63:]
        lines = {"stock": [r2(x / a.iloc[0] * 100) for x in a]}
        n = nifty_hist()
        if n is not None:
            nn = n.reindex(a.index).ffill()
            lines["nifty"] = [r2(x / nn.iloc[0] * 100) for x in nn]
        if sser is not None:
            ss = sser.reindex(a.index).ffill().bfill()
            lines["sector"] = [r2(x / ss.iloc[0] * 100) for x in ss]
        lines["m3"] = {"stock": r2(ret(h["close"], 63)), "sector": r2(sector_move(sec, "m3")) if sec else None, "nifty": r2(nifty_move("m3"))}
    news = N.stock(sym, c.get("name"))
    vol60 = st.get("vol60")
    return {"sym": sym, "row": row, "cls": {k: c.get(k) for k in ("sector", "industry", "basic", "pe", "mcap", "delivery", "high52", "low52", "name")},
            "sector": sec and {"name": sec["name"], "index": sec["index"]}, "peers": sorted(peers, key=lambda p: -(p["pct"] or -99)),
            "against": against, "rank": rank, "lines": lines, "news": news,
            "volume": {"today": q.get("volume"), "normal": r2(vol60), "ratio": r2((q.get("volume") or 0) / vol60) if vol60 else None},
            "delivery": c.get("delivery"), "dma50": r2(st.get("dma50")),
            "chart": rel_chart(h["close"] if h is not None else None, nifty_hist(), news)}


def search(q):
    q = q.strip().upper()
    out = [{"type": "sector", "name": s["name"]} for s in U.SECTORS if q in s["name"].upper()]
    for s, c in U.U["cls"].items():
        if q in s or q in (c.get("name") or "").upper():
            out.append({"type": "stock", "sym": s, "name": c.get("name"), "pct": L.price(s).get("pct")})
        if len(out) > 14:
            break
    return out


def chart(sym, tf):
    rng = {"1m": "5d", "5m": "1mo", "15m": "60d", "60m": "2y", "1d": "5y"}[tf]
    y = sym if sym.startswith("^") else sym + ".NS"
    d = M.get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(y)}?interval={tf}&range={rng}")["chart"]["result"][0]
    o = d["indicators"]["quote"][0]
    df = pd.DataFrame({k: o[k] for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(d["timestamp"], unit="s", utc=True)).dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)
    rd = smc.read(df, 2 if tf in ("1m", "5m") else 3) if len(df) >= 30 else None
    m = d["meta"]
    return {"sym": sym, "tf": tf, "ltp": m.get("regularMarketPrice"), "prev": m.get("chartPreviousClose"), "time": m.get("regularMarketTime"),
            "candles": [{"time": int(t.timestamp()), "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume} for t, r in df.iterrows()],
            "read": rd}


def tick(syms):
    """Live prices for the symbols on screen (the page polls this every 3 seconds)."""
    out = {s: {"ltp": r2(L.price(s).get("ltp")), "pct": r2(L.price(s).get("pct"))} for s in syms}
    ix = {s["name"]: {"v": r2(L.index_value(s["index"]).get("pct")), "last": L.index_value(s["index"]).get("last")} for s in U.SECTORS}
    n = L.index_value("NIFTY 50")
    return {"stocks": out, "sectors": ix, "nifty": {"v": r2(n.get("pct")), "last": n.get("last")}, "time": L.S["time"], "market": M.market_status()}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT / "web"), **k)

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            # Smart Money
            if u.path == "/api/sm/live":
                return self.send_json(F.snapshot(paths=q.get("paths", "1") == "1"))
            if u.path == "/api/sm/chart":
                return self.send_json(F.chart(q.get("sym", "NIFTY").upper(), q.get("tf", "5m")))
            if u.path == "/api/sm/quote":
                return self.send_json(F.quote(q.get("sym", "NIFTY").upper()))
            # Research note under the stock chart
            if u.path == "/api/research":
                sym = q.get("sym", "").upper()
                if not sym or sym in {"NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"}:
                    return self.send_json({"error": "indices have no company report"})
                return self.send_json(RS.report(sym, fresh=q.get("fresh") == "1"))
            # Strategy Lab
            if u.path == "/api/lab/trades":
                sym = q.get("sym", "NIFTY").upper()
                if sym not in LV.B.INDICES:          # stocks: swing system only
                    return self.send_json(SW.trades_for(sym, q.get("strat") or None, q.get("exit") or None))
                if q.get("tf") == "1d":
                    return self.send_json({"trades": [], "note": "Index: the intraday system runs on the 5m chart."})
            if u.path == "/api/lab/swing":
                return self.send_json(SW.summary())
            if u.path == "/api/lab/trades":
                return self.send_json(LV.chart_trades(q.get("sym", "NIFTY").upper(), q.get("tf", "5m"), q.get("strat") or None,
                                                      q.get("trend") or None, q.get("exit") or None))
            if u.path == "/api/lab/summary":
                p = M.DATA / "lab_summary.json"
                return self.send_json({**json.loads(p.read_text()), "list": list(STRATEGIES)} if p.exists() else {"error": "run python -m lab.report"})
            if u.path == "/api/lab/journal":
                return self.send_json(LV.journal())
            # Sector Map
            if u.path == "/api/status":
                return self.send_json({"ready": bool(L.S["indices"]) and bool(U.U["cls"]), "progress": U.U["progress"], "time": L.S["time"],
                                       "classified": len(U.U["cls"]), "history": len(U.U["hist"])})
            if u.path == "/api/horizon":
                return self.send_json(horizon(q.get("range", "today")))
            if u.path == "/api/sector":
                return self.send_json(sector_view(q.get("name", "Energy"), q.get("range", "today")))
            if u.path == "/api/stock":
                return self.send_json(stock_view(q.get("sym", "RELIANCE")))
            if u.path == "/api/search":
                return self.send_json(search(q.get("q", "")))
            if u.path == "/api/chart":
                return self.send_json(chart(q.get("sym", "RELIANCE").upper(), q.get("tf", "5m")))
            if u.path == "/api/tick":
                return self.send_json(tick([s for s in q.get("syms", "").split(",") if s][:150]))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self.send_json({"error": str(e)}, 500)
        return super().do_GET()


def boot():
    log = lambda m: print(m, flush=True)
    U.U["list"] = U.nifty500()
    if U.CLS.exists():
        U.U["cls"] = json.loads(U.CLS.read_text())
    threading.Thread(target=L.run, kwargs={"log": log}, daemon=True).start()
    threading.Thread(target=F.run, kwargs={"log": log}, daemon=True).start()
    threading.Thread(target=LV.run, kwargs={"log": log}, daemon=True).start()
    threading.Thread(target=SW.run, kwargs={"log": log}, daemon=True).start()


    def slow():
        while True:
            try:
                U.build_classification(log)
                U.build_history(log)
            except Exception as e:
                log(f"universe: {e}")
            time.sleep(3600)
    threading.Thread(target=slow, daemon=True).start()

    def warm():
        while True:
            try:
                N.warm([s["name"] for s in U.SECTORS])
                _C["mk"] = (time.time(), N.market())
            except Exception as e:
                log(f"news: {e}")
            time.sleep(600)
    threading.Thread(target=warm, daemon=True).start()


if __name__ == "__main__":
    boot()
    print(f"Market LAB: http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
