# Market Lab: an NSE market dashboard you run on your own computer

A free, open-source dashboard for the Indian stock market. It runs on **your** computer, fetches **public**
data, and does every calculation locally. Nobody sends you tips:
you look at the market and decide for yourself.

> **Educational software, not investment advice.** The authors are not SEBI-registered investment advisers or
> research analysts. Patterns, backtests and research notes are produced by fixed rules from public data and may be
> wrong or delayed. Backtests are not promises: past results do not predict future returns. You are responsible for
> every trade you take.

## Quick start (no coding needed)

**You do not need Claude or any AI subscription.** No paid account of any kind. This is ordinary free software that runs on your own computer.

**What you need**
- Any Windows, Mac or Linux computer with an internet connection
- **Python** (free) from [python.org/downloads](https://www.python.org/downloads/)
  - On Windows, tick **"Add python.exe to PATH"** at the bottom of the installer
- About **1 GB** of free disk space, and **10-20 minutes** for the first setup

**Windows**
1. Click the green **Code** button at the top of this page, then **Download ZIP**
2. Right-click the ZIP file and choose **Extract All**
3. Open the extracted folder and double-click **`install.bat`**. The first run takes 10-20 minutes; keep the window open
4. When it finishes, double-click **`run.bat`**. The dashboard opens in your browser at **http://localhost:8810**

**Mac**
1. Click **Code**, then **Download ZIP**, then double-click the ZIP to unpack it
2. Open **Terminal** (press Cmd+Space, type "Terminal") and run:
   ```bash
   cd ~/Downloads/market-lab-main
   bash install.sh
   ```
3. After that, start it any time with:
   ```bash
   cd ~/Downloads/market-lab-main && bash run.sh
   ```

The dashboard keeps running while the window (or Terminal) stays open, and prices update on their own.
To refresh the history and re-run the backtest, run `install.bat` (Windows) or `bash install.sh` (Mac) again.

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

## License
MIT. See [LICENSE](LICENSE).
