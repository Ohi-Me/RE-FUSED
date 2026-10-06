import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np
from refused_gate import data, ladder
t = time.time()
ds = data.nyiso_load("dev")
print("load", round(time.time() - t, 1), "s", ds["df"].shape, ds["df"].split.value_counts().to_dict(), flush=True)
fr = ds["df"].assign(y_t=ds["df"].y, B=ds["df"]["iso_fc"])
cfg = ladder.LadderConfig(n_bag=2)
out, diag = ladder.run_ladder(fr, ds["feats"], ["hour_block", "hour"], cfg, seed=0, verbose=True)
print(ladder.score(out).round(4).to_string())
print(json.dumps({k: (v if k != "part" else {kk: {a: round(b, 3) if isinstance(b, float) else b for a, b in vv.items()} for kk, vv in v.items()}) for k, v in diag.items()}, indent=1, default=str)[:3000])
