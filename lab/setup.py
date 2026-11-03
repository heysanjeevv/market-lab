"""First-time setup: download the daily price history and run the swing backtest on YOUR computer, so the
Swing Lab, the stock radar and the charts' swing trades have something to show. Takes about 10-20 minutes.
    python -m lab.setup              # daily data + swing backtest (enough for everything except the Nifty intraday layer)
    python -m lab.setup --intraday   # also 5-minute / hourly data and the intraday research (another ~40 minutes)
Run it again any time to refresh."""
import sys
import time

from . import bars as B


def main():
    t0 = time.time()
    intraday = "--intraday" in sys.argv
    syms = list(B.INDICES) + B.universe()
    print(f"1/3  downloading daily history for {len(syms)} stocks and indices (polite, one at a time)...", flush=True)
    B.fetch_all(syms, tfs=("1d",) + (("5m", "60m") if intraday else ()), fresh_hours=12, log=lambda m: print("     " + m, flush=True))
    print("2/3  running the swing backtest...", flush=True)
    from . import swing
    t = swing.run_all()
    print(f"     {len(t):,} simulated trades", flush=True)
    print("3/3  building the Swing Lab summary...", flush=True)
    from . import swing_live
    swing_live.build_summary()
    if intraday:
        print("     intraday research (5-minute and hourly grids)...", flush=True)
        from . import engine as E, report
        for tf in ("5m", "60m"):
            ss = sorted(p.stem for p in (B.ROOT / tf).glob("*.pkl"))
            E.run_all(ss, tf).to_pickle(B.ROOT.parent / f"trades_{tf}.pkl")
        report.build()
    print(f"done in {round((time.time() - t0) / 60)} min. Start the dashboard with ./run.sh (or run.bat on Windows).", flush=True)


if __name__ == "__main__":
    main()
