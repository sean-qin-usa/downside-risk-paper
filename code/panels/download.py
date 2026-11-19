# download.py -- yfinance adjusted closes for the panels of code/panels/panels.json into data/panels/<panel>/<ticker>.csv
# (not committed). Tolerant of failures; coverage is written to results/panels/download_coverage.json.
import json, os, sys, time
import pandas as pd, yfinance as yf
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."); P=json.load(open(os.path.join(ROOT,"code","panels","panels.json"))); cov={}
for name,cfg in P.items():
    d=os.path.join(ROOT,"data","panels",name); os.makedirs(d,exist_ok=True); cov[name]={}
    tk=cfg["tickers"]
    for i in range(0,len(tk),40):
        chunk=tk[i:i+40]
        try: df=yf.download(chunk,start=cfg["start"],end="2026-11-14",auto_adjust=True,progress=False,group_by="ticker",threads=True)
        except Exception as e: print("chunk failed",name,i,e); continue
        for t in chunk:
            try:
                s=df[t]["Close"].dropna() if len(chunk)>1 else df["Close"].dropna()
                s=s[s>0]
                if len(s)<50: cov[name][t]=0; continue
                s.to_frame("Close").to_csv(os.path.join(d,t.replace("=","_")+".csv")); cov[name][t]=int(len(s))
            except Exception as e: cov[name][t]=0
        time.sleep(1)
    print(name,"listed",len(tk),"downloaded",sum(v>0 for v in cov[name].values()),flush=True)
os.makedirs(os.path.join(ROOT,"results","panels"),exist_ok=True); json.dump(cov,open(os.path.join(ROOT,"results","panels","download_coverage.json"),"w"),indent=1)
