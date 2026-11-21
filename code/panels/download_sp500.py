# download_sp500.py -- S&P 500 constituents with daily returns and market values, and the index's own return (CRSP, WRDS).
#   python code/panels/download_sp500.py <first date> <last date>
# Tables: crsp.dsp500list (membership spells), crsp.dsf (return, price, shares, permco), crsp.dsp500 (index returns).
# Output, in data/panels/sp500/ (licensed; not committed):
#   daily.pkl    permno, permco, date, ret (percent), cap (price times shares, thousands), member (in the index that day)
#   index.csv    date, vwretd, sprtrn (percent)
#   permnos.csv  permno, date, ret -- the list that code/panels/download_ibes_dates.py reads
import os, sys, numpy as np, pandas as pd, wrds
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."); d0, d1 = sys.argv[1:3]; out = os.path.join(ROOT, "data", "panels", "sp500"); os.makedirs(out, exist_ok=True)
user = [l.strip().split(":")[3] for l in open(os.path.expanduser("~/.pgpass")) if "wrds" in l.split(":")[0]][0]; db = wrds.Connection(wrds_username=user)
L = db.raw_sql("select permno, start, ending from crsp.dsp500list where ending >= '%s' and start <= '%s'" % (d0, d1), date_cols=["start", "ending"]); perm = sorted(L.permno.astype(int).unique())
D = db.raw_sql("select permno, permco, date, ret, abs(prc) * shrout as cap from crsp.dsf where date between '%s' and '%s' and permno in (%s)" % (d0, d1, ",".join(map(str, perm))), date_cols=["date"])
I = db.raw_sql("select caldt as date, vwretd, sprtrn from crsp.dsp500 where caldt between '%s' and '%s'" % (d0, d1), date_cols=["date"]); db.close()
D["permno"] = D.permno.astype(int); D["permco"] = D.permco.astype(int); D["ret"] = D.ret.astype(float) * 100.0; D["cap"] = D.cap.astype(float); D = D.sort_values(["permno", "date"]).reset_index(drop=True); D["member"] = False
for p, s, e in L.itertuples(index=False): D.loc[(D.permno == int(p)) & (D.date >= s) & (D.date <= e), "member"] = True
D.to_pickle(os.path.join(out, "daily.pkl")); I[["vwretd", "sprtrn"]] = I[["vwretd", "sprtrn"]].astype(float) * 100.0; I.to_csv(os.path.join(out, "index.csv"), index=False)
D.groupby("permno").head(1)[["permno", "date", "ret"]].to_csv(os.path.join(out, "permnos.csv"), index=False)
print("%d permnos, %d firms, %d rows, %s to %s; members per day: median %d" % (len(perm), D.permco.nunique(), len(D), D.date.min().date(), D.date.max().date(), int(D[D.member].groupby("date").size().median())))
