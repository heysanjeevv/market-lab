"""Headlines for a sector or a stock (Google News, India edition), tagged as results, policy
or news, cached for 15 minutes per query."""
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from . import market as M

_CACHE = {}
RESULTS = re.compile(r"\b(q[1-4]|quarter|results?|profit|earnings|revenue|ebitda|net income|fy2\d)\b", re.I)
POLICY = re.compile(r"\b(rbi|sebi|government|govt|ministry|policy|budget|tariff|gst|duty|cabinet|regulat|ban|subsid|import|export curb|centre|minister)\b", re.I)
QUERY = {"Railways": "railway stocks OR IRCTC OR RVNL OR IRFC", "Energy": "power stocks OR oil gas stocks india OR coal india OR ntpc",
         "Infra": "infrastructure stocks india OR L&T order", "Cons Durables": "consumer durables stocks india OR dixon OR voltas OR havells",
         "Metal": "metal stocks india OR tata steel OR hindalco OR jsw steel", "Tourism": "hotel stocks india OR indigo OR tourism stocks",
         "Defence": "defence stocks india OR HAL OR BEL OR mazagon dock", "Auto": "auto stocks india OR maruti OR tata motors sales",
         "Chemicals": "chemical stocks india", "PSU Bank": "psu bank stocks OR SBI OR bank of baroda", "Cap Markets": "BSE share OR CDSL OR MCX OR capital market stocks",
         "Realty": "realty stocks india OR DLF OR godrej properties", "FMCG": "fmcg stocks india OR HUL OR ITC OR nestle india",
         "Fin Services": "nbfc stocks india OR bajaj finance", "Healthcare": "hospital stocks india OR apollo hospitals OR max healthcare",
         "IT": "IT stocks india OR infosys OR tcs OR wipro", "Bank": "bank stocks india OR HDFC bank OR ICICI bank", "Pharma": "pharma stocks india OR sun pharma OR usfda",
         "Media": "media stocks india OR zee OR sun tv OR PVR", "Cement": "cement stocks india OR ultratech OR ambuja"}


def tag(title):
    return "RESULTS" if RESULTS.search(title) else "POLICY" if POLICY.search(title) else "NEWS"


def search(q, days=30, limit=20):
    key = (q, days)
    if key in _CACHE and time.time() - _CACHE[key][0] < 900:
        return _CACHE[key][1]
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(f"{q} when:{days}d") + "&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        root = ET.fromstring(M.get(url))
    except Exception:
        return _CACHE.get(key, (0, []))[1]
    out, seen, cutoff = [], set(), datetime.now(timezone.utc) - timedelta(days=days)
    for it in root.iter("item"):
        src, title = it.findtext("source", ""), it.findtext("title", "")
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        k = title.lower()[:60]
        if k in seen:
            continue
        seen.add(k)
        try:
            t = parsedate_to_datetime(it.findtext("pubDate"))
        except Exception:
            continue
        if t < cutoff:
            continue
        out.append({"time": t.isoformat(), "title": title, "source": src, "link": it.findtext("link", ""), "tag": tag(title)})
    out.sort(key=lambda x: x["time"], reverse=True)
    _CACHE[key] = (time.time(), out[:limit])
    return out[:limit]


def sector(name, wait=True):
    q = QUERY.get(name, name + " stocks india")
    if not wait:
        return _CACHE.get((q, 30), (0, []))[1]      # whatever the background warmer already has
    return search(q)


def warm(names):
    """Keep every sector's and the market's headlines fresh in the background."""
    for n in names:
        sector(n)
        time.sleep(0.5)
    market()


def stock(sym, company):
    short = re.sub(r"\b(limited|ltd\.?|corporation|corp\.?|india|industries)\b", "", company or sym, flags=re.I).strip(" .,")
    return search(f'"{short}" OR {sym}', days=60, limit=12)


def market():
    return search("sensex OR nifty OR \"stock market\" india", days=30, limit=30)
