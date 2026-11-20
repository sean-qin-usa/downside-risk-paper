# download_optionm_earnings.py -- option-implied earnings jump variance for the canonical CRSP panel (OptionMetrics via WRDS).
#
# For every reaction day of every name (data/panels/canon_calcache.pkl), the at-the-money implied volatility of the
# 30-day and 60-day standardized options at the previous close (mean of call and put). With one announcement inside
# both maturities, implied variance is  IV(T)^2 = s^2 + J^2 / T,  so the implied variance of the announcement return is
#     J^2 = (IV30^2 - IV60^2) / (1/T30 - 1/T60),   T in years   (Dubinsky, Johannes, Kaeck and Seeger 2019).
# Needs a WRDS login file (~/.pgpass) created by the account holder. Licensed data: the output stays in data/ and is
# not committed.   python code/panels/download_optionm_earnings.py
import os, sys, time
import numpy as np, pandas as pd, wrds
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."); t0 = time.time()
user = [l.strip().split(":")[3] for l in open(os.path.expanduser("~/.pgpass")) if "wrds" in l.split(":")[0]][0]; db = wrds.Connection(wrds_username=user)
A = pd.read_pickle(os.path.join(ROOT, "data", "panels", "canon_calcache.pkl")); A["permno"] = A["name"].astype(int); perm = sorted(A.permno.unique())
lk = db.raw_sql("select secid, permno, sdate, edate, score from wrdsapps_link_crsp_optionm.opcrsphist where permno in (%s)" % ",".join(map(str, perm)), date_cols=["sdate", "edate"]); lk = lk[lk.score <= 2]
print("linked permnos: %d of %d; secids %d" % (lk.permno.nunique(), len(perm), lk.secid.nunique()), flush=True); secs = ",".join(str(int(s)) for s in lk.secid.unique()); parts = []
for yr in range(2014, 2025):
    q = db.raw_sql("select secid, date, days, avg(impl_volatility) as iv from optionm.stdopd%d where secid in (%s) and days in (30, 60) and impl_volatility > 0 group by secid, date, days" % (yr, secs), date_cols=["date"]); parts.append(q); print(yr, len(q), "%.0fs" % (time.time() - t0), flush=True)
db.close(); O = pd.concat(parts); O["iv"] = O["iv"].astype(float); O = O.pivot_table(index=["secid", "date"], columns="days", values="iv").reset_index(); O.columns = ["secid", "date", "iv30", "iv60"]
O = O.merge(lk[["secid", "permno", "sdate", "edate"]], on="secid"); O = O[(O.date >= O.sdate) & (O.date <= O.edate)].sort_values(["permno", "date"])
out = []
for p, g in A.groupby("permno"):
    d = g.date.values; r = g.date.values[(g.r0 == 1).values]; prev = d[np.maximum(np.searchsorted(d, r) - 1, 0)]; o = O[O.permno == p].set_index("date")[["iv30", "iv60"]]; o = o[~o.index.duplicated()].reindex(prev)
    out.append(pd.DataFrame({"name": str(p), "date": r, "iv30": o.iv30.values, "iv60": o.iv60.values}))
E = pd.concat(out, ignore_index=True); E["J2_opt"] = 1e4 * np.maximum(E.iv30 ** 2 - E.iv60 ** 2, 0) / (365 / 30 - 365 / 60)   # percent squared, as the returns
# The same from the option chain itself (added after the standardized options proved too crude): at the previous close,
# the two nearest expiries on or after the reaction day; for each, the mean implied volatility of the call and the put
# whose delta is nearest 0.5. iv1, T1 (years) are the nearest expiry; J2_ts uses both expiries as above.
db = wrds.Connection(wrds_username=user); ev = E[["name", "date"]].copy(); ev["permno"] = ev["name"].astype(int); ev["rdate"] = ev["date"]
prevmap = {}
for p, g in A.groupby("permno"):
    d = g.date.values; r = d[(g.r0 == 1).values]; prevmap.update({(p, pd.Timestamp(x)): pd.Timestamp(y_) for x, y_ in zip(r, d[np.maximum(np.searchsorted(d, r) - 1, 0)])})
ev["pdate"] = [prevmap[(p, pd.Timestamp(r))] for p, r in zip(ev.permno, ev.rdate)]; ev = ev.merge(lk[["secid", "permno", "sdate", "edate"]], on="permno"); ev = ev[(ev.pdate >= ev.sdate) & (ev.pdate <= ev.edate)]; ch = []
for yr in range(2014, 2025):
    e = ev[ev.pdate.dt.year == yr]
    if not len(e): continue
    vals = ",".join("(%d,'%s'::date)" % (int(s_), str(d_.date())) for s_, d_ in zip(e.secid, e.pdate))
    ch.append(db.raw_sql("select o.secid, o.date as pdate, o.exdate, o.cp_flag, o.delta, o.impl_volatility as iv from optionm.opprcd%d o join (values %s) as e(secid,date) on o.secid=e.secid and o.date=e.date where o.impl_volatility is not null and abs(o.delta) between 0.3 and 0.7 and o.exdate - o.date between 1 and 200" % (yr, vals), date_cols=["pdate", "exdate"]))
db.close(); ch = pd.concat(ch); ch["iv"] = ch["iv"].astype(float); ch["dist"] = (ch["delta"].astype(float).abs() - 0.5).abs(); ch = ch.sort_values("dist").groupby(["secid", "pdate", "exdate", "cp_flag"], as_index=False).first()
ch = ch.groupby(["secid", "pdate", "exdate"], as_index=False).agg(iv=("iv", "mean"), legs=("iv", "size")); ch = ch.merge(ev[["secid", "pdate", "rdate", "name"]], on=["secid", "pdate"]); ch = ch[ch.exdate >= ch.rdate].sort_values(["name", "rdate", "exdate"]); ch["T"] = (ch.exdate - ch.pdate).dt.days / 365.0; ch["k"] = ch.groupby(["name", "rdate"]).cumcount()
w1 = ch[ch.k == 0].set_index(["name", "rdate"])[["iv", "T"]]; w2 = ch[ch.k == 1].set_index(["name", "rdate"])[["iv", "T"]]; key = pd.MultiIndex.from_arrays([E["name"], E["date"]])
E["iv1"] = w1.iv.reindex(key).values; E["T1"] = w1["T"].reindex(key).values; iv2 = w2.iv.reindex(key).values; T2 = w2["T"].reindex(key).values
E["J2_ts"] = 1e4 * np.maximum(E.iv1 ** 2 - iv2 ** 2, 0) / (1 / E.T1 - 1 / T2); print("from the chain: nearest expiry for %d of %d reaction days, two expiries for %d; median days to the nearest expiry %.0f" % (E.iv1.notna().sum(), len(E), E.J2_ts.notna().sum(), 365 * E.T1.median()))
E.to_pickle(os.path.join(ROOT, "data", "panels", "optionm_earnings_canon.pkl")); print("reaction days %d, with both maturities %d (%.0f%%), implied jump sd median %.2f%%" % (len(E), E.J2_opt.notna().sum(), 100 * E.J2_opt.notna().mean(), np.sqrt(E.J2_opt.median())))
