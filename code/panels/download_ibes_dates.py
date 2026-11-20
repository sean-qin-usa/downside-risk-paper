# download_ibes_dates.py -- quarterly earnings announcement dates and times from I/B/E/S for a list of CRSP permnos (WRDS).
#   python code/panels/download_ibes_dates.py <csv with permno,date,ret> <tag> <first date> <last date>
# Link: wrdsapps.ibcrsphist (I/B/E/S ticker to permno, score 1 or 2). Actuals: ibes.actu_epsus, quarterly EPS, US file.
# Output: data/panels/earnings_ibes_<tag>/<permno>.csv with one timestamp per announcement (Eastern). Licensed; not committed.
import os, sys, numpy as np, pandas as pd, wrds
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."); src, tag, d0, d1 = sys.argv[1:5]; out = os.path.join(ROOT, "data", "panels", "earnings_ibes_" + tag); os.makedirs(out, exist_ok=True)
perm = sorted(pd.read_csv(src, usecols=["permno"]).permno.unique()); user = [l.strip().split(":")[3] for l in open(os.path.expanduser("~/.pgpass")) if "wrds" in l.split(":")[0]][0]; db = wrds.Connection(wrds_username=user)
lk = db.raw_sql("select ticker, permno, sdate, edate, score from wrdsapps.ibcrsphist where permno in (%s) and score <= 2" % ",".join(str(int(p)) for p in perm), date_cols=["sdate", "edate"])
tk = "','".join(sorted(lk.ticker.unique())); A = db.raw_sql("select ticker, pends, anndats, anntims from ibes.actu_epsus where measure='EPS' and pdicity='QTR' and anndats between '%s' and '%s' and ticker in ('%s')" % (d0, d1, tk), date_cols=["pends", "anndats"]); db.close()
A = A.merge(lk, on="ticker"); A = A[(A.anndats >= A.sdate) & (A.anndats <= A.edate)].drop_duplicates(["permno", "anndats"]); n = 0; cnt = []
for p, g in A.groupby("permno"):
    t = g.anntims.astype(str).str.slice(0, 8).where(g.anntims.notna(), "16:00:00"); s = sorted(set(g.anndats.dt.strftime("%Y-%m-%d") + " " + t)); pd.Series(s, name="announcement").to_csv(os.path.join(out, "%d.csv" % int(p)), index=False); n += 1; cnt.append(len(s))
print("%s: %d permnos asked, %d with a record, median announcements %d, total %d" % (tag, len(perm), n, int(np.median(cnt)), int(np.sum(cnt))))
