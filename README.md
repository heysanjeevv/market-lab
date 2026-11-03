# Market Lab: an NSE market dashboard you run on your own computer

A free, open-source dashboard for the Indian stock market. It runs on **your** computer, fetches **public**
data, and does every calculation locally. Nobody sends you tips:
you look at the market and decide for yourself.

> **Educational software, not investment advice.** The authors are not SEBI-registered investment advisers or
> research analysts. Patterns, backtests and research notes are produced by fixed rules from public data and may be
> wrong or delayed. Backtests are not promises: past results do not predict future returns. You are responsible for
> every trade you take.

## Install karna kitna aasaan hai? (Easy install, Hinglish)

**Claude ya koi AI nahi chahiye.** Na Claude Pro, na koi paid account. Yeh ek normal free software hai jo aapke PC par chalta hai.

**Kya chahiye:**
- Windows ya Mac (ya Linux) wala koi bhi PC/laptop, aur internet
- **Python** (free): [python.org/downloads](https://www.python.org/downloads/) se download kijiye
  - Windows par install karte waqt neeche **"Add python.exe to PATH"** zaroor tick kijiye
- Lagbhag **1 GB** khaali jagah, aur pehli baar **10–20 minute**

**Windows par:**
1. Is page par upar hare **Code** button par click kijiye, phir **Download ZIP**
2. ZIP file par right-click, phir **Extract All**
3. Folder ke andar **`install.bat`** par double-click kijiye. Pehli baar 10–20 minute lagenge, window band mat kijiye
4. Ho jaane ke baad **`run.bat`** par double-click kijiye. Browser mein dashboard khul jayega: **http://localhost:8810**

**Mac par:**
1. **Code** button, phir **Download ZIP**, phir ZIP par double-click karke kholiye
2. **Terminal** kholiye (Cmd+Space dabaiye, "Terminal" likhiye) aur yeh chalaiye:
   ```bash
   cd ~/Downloads/market-lab-main
   bash install.sh
   ```
3. Ho jaane ke baad, har baar chalane ke liye:
   ```bash
   cd ~/Downloads/market-lab-main && bash run.sh
   ```

Bas itna hi. Jab tak window ya Terminal khula hai, dashboard chalta rahega. Data apne aap update hota hai.
Latest data aur naya backtest chahiye to `install.bat` (Windows) ya `bash install.sh` (Mac) dobara chala dijiye.

## What's inside
- **Smart Money:** every F&O stock's move since the open, the waterfall of strongest and weakest stocks, sector chips,
  live tiles, and smart-money structure (break of structure, order blocks, fair-value gaps, liquidity sweeps).
- **Sector Map:** which sectors are beating Nifty today, this week and this month; drill into a sector and its stocks.
- **Charts:** 1-minute to daily candles with the smart-money layers. Stock charts open on the daily view and show
  the swing rules' past signals with their stop and target boxes (you can switch this layer off).
- **Research note under every stock chart:** what the company does in plain words, results for the last quarters
  (from NSE filings), valuation against its sector, peers, which parts of the economy move it, and a strengths /
  weaknesses list built from the numbers.
- **Swing Lab:** a backtest of 23 swing strategies on 5 years of daily data **that you run yourself**, a radar of
  stocks whose pattern is forming, and what kind of stocks the rules worked best in, on your data.

## Install (once)
You need **Python 3.10 or newer** ([python.org/downloads](https://www.python.org/downloads/); on Windows tick
"Add Python to PATH").

**Mac / Linux**
```bash
git clone https://github.com/heysanjeevv/market-lab.git   # or: Code > Download ZIP
cd market-lab
bash install.sh
```

**Windows:** download the repository as ZIP (green *Code* button, *Download ZIP*), unzip it, and double-click
`install.bat`.

The installer creates a private Python environment inside the folder, installs three libraries (numpy, pandas,
tzdata), downloads about 5 years of daily prices for ~520 NSE stocks and indices, and runs the
backtest. It takes 10-20 minutes the first time.

## Run
- Mac / Linux: `bash run.sh`
- Windows: double-click `run.bat`

Then open **http://localhost:8810** (it opens by itself). Keep the window running while you use it.
Refresh the data and backtest any time with `.venv/bin/python -m lab.setup` (Windows: `.venv\Scripts\python -m lab.setup`).

## Data sources
NSE's public website (index members, quotes, company filings), Yahoo Finance (price history), Google News RSS
(headlines) and Wikipedia (company summaries), each fetched politely from your own computer for your own use. Check each source's terms before using the data for anything else.

---

## हिंदी में
यह एक मुफ़्त, open-source NSE डैशबोर्ड है जो **आपके अपने कंप्यूटर** पर चलता है। डेटा public sources से आता है और सारा हिसाब आपके कंप्यूटर पर होता है। यह **निवेश सलाह नहीं** है: patterns और backtests
नियमों से बनते हैं, गलत भी हो सकते हैं, और हर trade की ज़िम्मेदारी आपकी है।

**Install:** Python 3.10+ install कीजिए। Mac/Linux पर `./install.sh` चलाइए, Windows पर ZIP download करके
`install.bat` पर double-click कीजिए (पहली बार 10-20 मिनट)। फिर `run.sh` / `run.bat` चलाइए और
**http://localhost:8810** खोलिए।

## License
MIT. See [LICENSE](LICENSE).
