"""A one-page, plain-language research note for any NSE stock: what the company does, how its results are
moving, its finances, valuation against peers, the industry and the economy around it, a SWOT and an
overall score out of 100. Built from NSE's own filings and quote page, Wikipedia and a few market prices.
Rules-based (no opinions are invented); cached for 12 hours per stock."""
import json
import re
import time
import urllib.parse

import numpy as np
import pandas as pd

from sm import market as M
from sx import universe as U
from . import bars as B
from . import industries as I
from . import swing as S

DIR = M.DATA / "research"
DIR.mkdir(exist_ok=True)
NSE = "https://www.nseindia.com/api"
MACRO_SYMS = {"oil": "BZ=F", "inr": "USDINR=X", "metals": "HG=F", "us": "^GSPC"}
_MACRO = {"t": 0, "v": {}}


def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def quarters(sym):
    """Last 8 quarters from NSE's integrated filings, in Rs crore (NSE reports Rs lakh). Consolidated when filed."""
    rows = M.get_json(f"{NSE}/NextApi/apiClient/GetQuoteApi?functionName=getIntegratedFilingData&symbol={urllib.parse.quote(sym)}", 20) or []
    by = {}
    for r in rows:
        q = r.get("gfrQuaterEnded")
        if not q:
            continue
        kind = r.get("gfrConsolidated") or ""
        cur = by.get(q)
        if cur is None or (kind == "Consolidated" and cur["kind"] != "Consolidated" and _num(r.get("gfrNetProLoss")) is not None):
            by[q] = {"q": q, "kind": kind, "income": _num(r.get("gfrTotalIncome")), "pbt": _num(r.get("gfrProBefTax")),
                     "profit": _num(r.get("gfrNetProLoss")), "eps": _num(r.get("gfrErnPerShare"))}
    out = []
    for q, v in by.items():
        try:
            d = pd.Timestamp(q)
        except Exception:
            continue
        out.append({**v, "date": str(d.date()), "label": d.strftime("%b %y"),
                    **{k: (v[k] / 100 if v[k] is not None else None) for k in ("income", "pbt", "profit")}})
    return sorted(out, key=lambda x: x["date"])[-8:]


def balance(sym):
    """Debt/equity and interest cover from NSE's results-comparison filing (older but the only free source)."""
    try:
        d = M.get_json(f"{NSE}/results-comparision?symbol={urllib.parse.quote(sym)}", 20)
        r = (d.get("resCmpData") or [{}])[0]
        return {"de": _num(r.get("re_debt_eqt_rat")), "icr": _num(r.get("re_int_ser_cov")), "as_of": r.get("re_to_dt"),
                "interest": (_num(r.get("re_int_new")) or 0) / 100, "dep": (_num(r.get("re_depr_und_exp")) or 0) / 100}
    except Exception:
        return {}


def wiki(name):
    """Two or three sentences about the company from Wikipedia."""
    if not name:
        return None
    short = re.sub(r"\b(limited|ltd\.?)$", "", name.strip(), flags=re.I).strip()
    try:
        s = M.get_json("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=3&srsearch=" +
                       urllib.parse.quote(f'{short} company India'), 15)
        for hit in s.get("query", {}).get("search", []):
            title = hit["title"]
            words = [w for w in re.findall(r"[a-z]+", short.lower()) if len(w) > 2][:3]
            if not any(w in title.lower() for w in words):
                continue
            p = M.get_json("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")), 15)
            ext = p.get("extract") or ""
            if ext:
                sent = re.split(r"(?<=[.!?])\s+", ext)
                return {"text": " ".join(sent[:3]), "url": (p.get("content_urls") or {}).get("desktop", {}).get("page"), "title": title}
    except Exception:
        return None
    return None


def macro():
    """3-month change of crude, the dollar-rupee, copper and the S&P 500 (cached 6 hours)."""
    if time.time() - _MACRO["t"] < 6 * 3600 and _MACRO["v"]:
        return _MACRO["v"]
    out = {}
    for k, y in MACRO_SYMS.items():
        try:
            d = M.get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(y)}?interval=1d&range=6mo", 15)["chart"]["result"][0]
            c = [x for x in d["indicators"]["quote"][0]["close"] if x]
            out[k] = {"now": round(c[-1], 2), "chg3m": round((c[-1] / c[-64] - 1) * 100, 1) if len(c) > 64 else None}
        except Exception:
            pass
    _MACRO.update(t=time.time(), v=out)
    return out


def peers(sym, cls):
    me = cls.get(sym) or {}
    same = [(k, v) for k, v in cls.items() if v.get("basic") == me.get("basic") and k != sym]
    if len(same) < 3:
        same = [(k, v) for k, v in cls.items() if v.get("industry") == me.get("industry") and k != sym]
    same.sort(key=lambda kv: -(kv[1].get("mcap") or 0))
    return [{"sym": k, "name": (v.get("name") or k)[:28], "mcap_cr": round((v.get("mcap") or 0) / 1e7), "pe": v.get("pe")} for k, v in same[:6]]


def ret(sym, days):
    d = B.load(sym, "1d")
    if d is None or len(d) <= days:
        return None
    c = d.close.values
    return round(float((c[-1] / c[-1 - days] - 1) * 100), 1)


def sector_strength(sym, cls):
    me = cls.get(sym) or {}
    mates = [k for k, v in cls.items() if v.get("sector") == me.get("sector")]
    r = [x for x in (ret(k, 63) for k in mates[:40]) if x is not None]
    n = ret("NIFTY", 63)
    return {"sector": me.get("sector"), "sector_3m": round(float(np.median(r)), 1) if r else None, "nifty_3m": n,
            "rel": round(float(np.median(r)) - n, 1) if r and n is not None else None, "n": len(r)}


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def lin(v, a, b):
    """Map v from [a, b] to [0, 10]."""
    if v is None:
        return None
    return round(clip((v - a) / (b - a) * 10, 0, 10), 1)


def build(sym):
    q = U.quote(sym)
    try:
        cls = json.loads((M.DATA / "classification.json").read_text())
    except Exception:
        cls = {}
    cls.setdefault(sym, q)
    macro_name, basic = q.get("macro") or "", q.get("basic") or ""
    what, drivers = I.lookup(basic, macro_name)
    fin = quarters(sym)
    bal = balance(sym)
    wk = wiki(q.get("name"))
    mac = macro()
    sec = sector_strength(sym, cls)
    pr = peers(sym, cls)
    mcap_cr = (q.get("mcap") or 0) / 1e7
    fs = macro_name == "Financial Services"

    # ---------- results
    inc = [x["income"] for x in fin if x["income"] is not None]
    pat = [x["profit"] for x in fin if x["profit"] is not None]
    ttm_inc = sum(inc[-4:]) if len(inc) >= 4 else None
    ttm_pat = sum(pat[-4:]) if len(pat) >= 4 else None
    # growth: each recent quarter against the same quarter a year earlier, over whatever pairs NSE has
    byd = {x["date"][:7]: x for x in fin}
    pairs = []
    for x in fin[-4:]:
        d0 = pd.Timestamp(x["date"]) - pd.DateOffset(years=1)
        y = byd.get(str(d0.date())[:7])
        if y:
            pairs.append((x, y))
    def growth(key, pos=False):
        a = [(x[key], y[key]) for x, y in pairs if x[key] is not None and y[key] is not None]
        if not a:
            return None
        now, then = sum(u for u, _ in a), sum(v for _, v in a)
        return (now / then - 1) * 100 if then > 0 else None
    g_inc, g_pat = growth("income"), growth("profit")
    n_pairs = len(pairs)
    q_yoy = None
    if len(fin) >= 5 and fin[-1]["profit"] is not None and fin[-5]["profit"] and fin[-5]["profit"] > 0:
        q_yoy = (fin[-1]["profit"] / fin[-5]["profit"] - 1) * 100
    margin = ttm_pat / ttm_inc * 100 if ttm_inc and ttm_pat is not None else None
    norm = I.MARGIN_NORM.get(macro_name, 8)
    profit_q = sum(1 for p in pat if p > 0)
    pe, spe = q.get("pe"), q.get("sector_pe")
    if (not pe or pe <= 0) and ttm_pat and ttm_pat > 0 and mcap_cr:
        pe = mcap_cr / ttm_pat

    # ---------- price & trend (daily bars)
    tech = {}
    try:
        d = B.load(sym, "1d")
        if d is not None and len(d) > 260:
            f = S.features(d[d.volume > 0], S.nifty_table())
            i = f["n"] - 1
            tech = {"uptrend": bool(f["uptrend"][i]), "rs63": round(float(f["rs63"][i]), 1) if np.isfinite(f["rs63"][i]) else None,
                    "from_high": round(float((f["c"][i] / f["hh252"][i] - 1) * 100), 1) if np.isfinite(f["hh252"][i]) else None,
                    "atr_pct": round(float(f["atr"][i] / f["c"][i] * 100), 2), "ret1y": ret(sym, 250), "ret3m": ret(sym, 63)}
    except Exception:
        pass

    # ---------- economy: which way each driver is pushing right now
    econ = []
    tail = 0
    for fac, eff, why in drivers:
        reading = mac.get(fac) if fac in mac else None
        push = None
        if reading and reading.get("chg3m") is not None and eff != 0:
            ch = reading["chg3m"]
            thr = 1.5 if fac == "inr" else 3
            if abs(ch) >= thr:
                push = eff * (1 if ch > 0 else -1)
                tail += push
        elif reading and reading.get("chg3m") is not None and abs(reading["chg3m"]) >= 10:
            push, tail = 0, tail - .5          # a two-way driver moving this much is a risk either way
        econ.append({"factor": fac, "name": I.FACTOR_NAMES.get(fac, fac), "effect": eff, "why": why,
                     "reading": reading, "push": push})

    # ---------- scores (0-10 each)
    sc = {}
    size = lin(np.log10(max(mcap_cr, 1)), 2.5, 5.3)              # ~Rs 300 cr .. Rs 2 lakh cr
    cons = profit_q / max(len(pat), 1) * 10 if pat else None
    sc["Business"] = round(np.nanmean([x for x in (size, cons) if x is not None]), 1) if (size is not None or cons is not None) else None
    sc["Growth"] = round(np.nanmean([x for x in (lin(g_inc, -10, 25), lin(g_pat, -20, 40), lin(q_yoy, -30, 50)) if x is not None]), 1) if any(x is not None for x in (g_inc, g_pat, q_yoy)) else None
    sc["Profitability"] = lin(margin / norm if margin is not None else None, 0, 2)
    if fs:
        sc["Financial health"] = round(cons, 1) if cons is not None else None
    else:
        de = bal.get("de")
        de_s = None if de is None else (10 if de < .3 else 8 if de < .6 else 6 if de < 1 else 3.5 if de < 2 else 1)
        icr_s = None          # NSE's coverage field is unreliable; debt/equity and profit record decide
        sc["Financial health"] = round(np.nanmean([x for x in (de_s, icr_s, cons) if x is not None]), 1) if any(x is not None for x in (de_s, icr_s, cons)) else None
    if pe and pe > 0 and spe:
        r = pe / spe
        v = 9 if r <= .7 else 7.5 if r <= .9 else 6 if r <= 1.1 else 4.5 if r <= 1.4 else 3 if r <= 2 else 1.5
        if g_pat is not None and g_pat > 25 and r < 1.6:
            v = min(10, v + 1.5)
        sc["Valuation"] = v
    else:
        sc["Valuation"] = 2 if ttm_pat is not None and ttm_pat <= 0 else None
    ind = lin(sec.get("rel"), -10, 10)
    sc["Industry & economy"] = round(clip((ind if ind is not None else 5) + tail * 1.5, 0, 10), 1)
    if tech:
        m = (4 if tech.get("uptrend") else 0) + (lin(tech.get("from_high"), -30, 0) or 0) * .3 + (3 if (tech.get("rs63") or 0) > 0 else 0)
        sc["Momentum"] = round(clip(m, 0, 10), 1)
    W = {"Business": 15, "Growth": 20, "Profitability": 15, "Financial health": 15, "Valuation": 15, "Industry & economy": 10, "Momentum": 10}
    have = {k: v for k, v in sc.items() if v is not None}
    total = round(sum(have[k] * W[k] for k in have) / sum(W[k] for k in have) * 10) if have else None
    grade = None if total is None else ("Strong" if total >= 75 else "Good" if total >= 62 else "Average" if total >= 48 else "Weak" if total >= 35 else "Poor")

    # ---------- SWOT
    s_, w_, o_, t_ = [], [], [], []
    if g_pat is not None and g_pat > 20: s_.append(f"Profit is growing fast: up {g_pat:.0f}% on the same quarters a year earlier.")
    if g_inc is not None and g_inc > 15: s_.append(f"Sales are growing: up {g_inc:.0f}% on a year earlier.")
    if margin is not None and margin >= norm: s_.append(f"Keeps a healthy share of sales as profit ({margin:.1f}%, typical for its kind is ~{norm}%).")
    if pat and profit_q == len(pat) and len(pat) >= 6: s_.append(f"Made a profit in every one of the last {len(pat)} quarters.")
    if not fs and bal.get("de") is not None and bal["de"] < .3: s_.append(f"Very little debt in its last detailed filing ({bal.get('as_of') or 'date n/a'}, own business only).")
    if pr and mcap_cr and all(mcap_cr >= p["mcap_cr"] for p in pr): s_.append("The biggest company in its industry on the market.")
    if tech.get("rs63") and tech["rs63"] > 10: s_.append(f"The share has beaten Nifty by {tech['rs63']:.0f} points over 3 months.")
    if margin is not None and margin < norm / 2: w_.append(f"Thin profit margin ({margin:.1f}% of sales), so small cost changes hurt.")
    if g_pat is not None and g_pat < -10: w_.append(f"Profit fell {abs(g_pat):.0f}% against a year earlier.")
    if g_inc is not None and g_inc < -5: w_.append(f"Sales shrank {abs(g_inc):.0f}% against a year earlier.")
    if pat and profit_q < len(pat): w_.append(f"Made a loss in {len(pat) - profit_q} of the last {len(pat)} quarters.")
    if not fs and bal.get("de") is not None and bal["de"] > 1.5: w_.append(f"Carries a lot of debt (debt {bal['de']:.1f}× its own capital, filing of {bal.get('as_of') or 'n/a'}).")
    if pe and spe and pe > 1.5 * spe: w_.append(f"Expensive: {pe:.0f}× its yearly profit against ~{spe:.0f}× for its sector.")
    if sec.get("rel") is not None and sec["rel"] > 3: o_.append(f"Its sector ({sec['sector']}) has beaten Nifty by {sec['rel']:.0f} points over 3 months: money is flowing in.")
    for e in econ:
        if e["push"] == 1: o_.append(f"{e['name']} is moving in its favour ({e['why']}).")
        if e["push"] == -1: t_.append(f"{e['name']} is moving against it ({e['why']}).")
        if e["push"] == 0: t_.append(f"{e['name']} moved {e['reading']['chg3m']:+.0f}% in 3 months; big swings hurt it ({e['why']}).")
    if pe and spe and pe < .85 * spe and (g_pat or 0) > 0: o_.append(f"Cheaper than its sector ({pe:.0f}× vs ~{spe:.0f}× profit) while profit is still growing.")
    if tech.get("uptrend") and tech.get("from_high") is not None and tech["from_high"] > -8: o_.append("Share is in an uptrend and close to its 52-week high.")
    if any(e["factor"] == "reg" for e in econ): t_.append("Depends on government or regulator decisions (" + next(e["why"] for e in econ if e["factor"] == "reg") + ").")
    if any(e["factor"] == "china" for e in econ): t_.append("Cheap Chinese imports can cut its selling prices.")
    if sec.get("rel") is not None and sec["rel"] < -3: t_.append(f"Its sector has lagged Nifty by {abs(sec['rel']):.0f} points over 3 months.")
    if tech.get("atr_pct") and tech["atr_pct"] > 3.5: t_.append(f"The share swings a lot (about {tech['atr_pct']:.1f}% on a normal day).")
    if tech.get("from_high") is not None and tech["from_high"] < -30: t_.append(f"The share is {abs(tech['from_high']):.0f}% below its 52-week high.")

    # ---------- plain-language summary
    size_word = "very large" if mcap_cr >= 100000 else "large" if mcap_cr >= 50000 else "mid-sized" if mcap_cr >= 15000 else "small" if mcap_cr >= 3000 else "very small"
    lines = [f"{q.get('name') or sym}: {what}",
             f"It is a {size_word} company: the market values the whole business at about ₹{mcap_cr:,.0f} crore."]
    if ttm_inc:
        lines.append(f"In the last 12 months it brought in ₹{ttm_inc:,.0f} crore and kept ₹{ttm_pat:,.0f} crore as profit"
                     + (f", about ₹{margin:.1f} of profit on every ₹100 of sales." if margin is not None else "."))
    if g_inc is not None or g_pat is not None:
        lines.append(("Compared with the same quarters a year earlier, " if n_pairs < 4 else "Compared with the 12 months before, ") + ", ".join(x for x in (
            f"income is {'up' if g_inc >= 0 else 'down'} {abs(g_inc):.0f}%" if g_inc is not None else None,
            f"profit is {'up' if g_pat >= 0 else 'down'} {abs(g_pat):.0f}%" if g_pat is not None else None) if x) + ".")
    if pe and pe > 0:
        cmp_ = "" if not spe else (" — cheaper than" if pe < .9 * spe else " — more expensive than" if pe > 1.1 * spe else " — about the same as") + f" its sector (~{spe:.0f}×)"
        lines.append(f"At today's price you pay about {pe:.0f}× one year's profit{cmp_}.")
    out = {"sym": sym, "name": q.get("name"), "built": int(time.time()), "macro": macro_name, "sector": q.get("sector"),
           "industry": q.get("industry"), "basic": basic, "index": q.get("index"), "what": what, "summary": lines, "wiki": wk,
           "mcap_cr": round(mcap_cr), "pe": round(pe, 1) if pe else None, "sector_pe": spe, "delivery": q.get("delivery"),
           "high52": q.get("high52"), "low52": q.get("low52"), "ltp": q.get("ltp"),
           "results": fin, "ttm": {"income": ttm_inc, "profit": ttm_pat, "margin": round(margin, 1) if margin is not None else None,
                                   "g_inc": round(g_inc, 1) if g_inc is not None else None, "g_pat": round(g_pat, 1) if g_pat is not None else None,
                                   "q_yoy": round(q_yoy, 1) if q_yoy is not None else None, "norm": norm, "pairs": n_pairs, "profit_q": profit_q, "n_q": len(pat)},
           "balance": bal, "financial": fs, "peers": pr, "sector_strength": sec, "economy": econ, "tech": tech,
           "scores": sc, "weights": W, "total": total, "grade": grade,
           "swot": {"Strengths": s_ or ["Nothing stands out."], "Weaknesses": w_ or ["No obvious weak spot in the numbers."],
                    "Opportunities": o_ or ["No clear tailwind right now."], "Threats": t_ or ["No clear headwind right now."]},
           "sources": ["NSE quarterly filings & quote page", "Wikipedia", "Yahoo (crude, dollar-rupee, copper, S&P 500)"]}
    return out


def report(sym, fresh=False):
    p = DIR / f"{sym}.json"
    if not fresh and p.exists() and time.time() - p.stat().st_mtime < 12 * 3600:
        return json.loads(p.read_text())
    r = build(sym)
    p.write_text(json.dumps(r, default=float))
    return r
