# download_earnings.py -- earnings announcement dates for the names of a stock panel (yfinance, free), used by
# tools/r71_shock_response.py to label each large shock as an earnings shock or not.
#   python code/panels/download_earnings.py <panel>                    tickers of code/panels/panels.json
#   python code/panels/download_earnings.py canon <crsp_panel_chars.csv> [rows.npz]  tickers of the canonical CRSP panel
# Output: data/panels/earnings/<ticker>.csv (one timestamp per line, exchange time; not committed) and a coverage file.
import os, sys, json, time
import pandas as pd, yfinance as yf
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", ".."); OUT = os.path.join(ROOT, "data", "panels", "earnings"); os.makedirs(OUT, exist_ok=True)
what = sys.argv[1]
if what == "canon":   # optional third argument: a per-row dump (.npz with a permno field) that restricts the names to the panel's
    ch = pd.read_csv(sys.argv[2])
    if len(sys.argv) > 3:
        import numpy as np; ch = ch[ch["permno"].isin(set(np.unique(np.load(sys.argv[3])["permno"]).tolist()))]
    tickers = [str(t) for t in ch["ticker"].dropna().unique()]
else: tickers = json.load(open(os.path.join(HERE, "panels.json")))[what]["tickers"]
cov = {}
for i, tk in enumerate(tickers):
    f = os.path.join(OUT, tk.replace("=", "_").replace("/", "_") + ".csv")
    if os.path.exists(f): cov[tk] = sum(1 for _ in open(f)) - 1; continue
    idx = []
    for off in (0, 100):
        for attempt in range(3):
            try:
                e = yf.Ticker(tk).get_earnings_dates(limit=100, offset=off)
                if (e is None or len(e) == 0) and off == 0 and attempt < 2: time.sleep(4 + 6 * attempt); continue   # an empty first page is usually throttling
                idx += [] if e is None else [str(x) for x in e.index]; break
            except Exception as ex:
                if attempt == 2: print(tk, off, "failed:", str(ex)[:80], flush=True)
                time.sleep(2 + 3 * attempt)
        time.sleep(0.6)
    idx = sorted(set(idx)); cov[tk] = len(idx)
    if idx: pd.Series(idx, name="announcement").to_csv(f, index=False)
    if i % 25 == 0: print(i, tk, len(idx), flush=True)
json.dump({"n_tickers": len(tickers), "with_dates": sum(1 for v in cov.values() if v > 0), "counts": cov}, open(os.path.join(ROOT, "results", "panels", "earnings_coverage_%s.json" % what), "w"), indent=1)
print("done: %d of %d tickers have dates" % (sum(1 for v in cov.values() if v > 0), len(tickers)))
