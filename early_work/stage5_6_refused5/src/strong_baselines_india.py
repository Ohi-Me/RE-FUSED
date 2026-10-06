"""RE-FUSED-5 R6: strong baseline suite on the India daily panel (test 2024-01..2025-04).
Statistical: Naive, SeasonalNaive, AutoARIMA, AutoETS, AutoTheta, WindowAverage (statsforecast).
ML (direct level): XGBoost, LightGBM. ML (residual): XGB/LGBM on persistence residual.
Plus RE-FUSED-5 shrunk-gated corrector. All share the identical split, features and horizon."""
import pandas as pd, numpy as np, os, time, warnings, json
warnings.filterwarnings("ignore")
D = r"D:\\REFUSED5\data"; OUT = r"D:\\REFUSED5\results"
p = pd.read_parquet(os.path.join(D, "india_model_daily.parquet")).sort_values(["state_name","date"])
H = 1
res = []
t0 = time.time()
# ---------- statistical baselines via statsforecast ----------
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoARIMA, AutoETS, AutoTheta, WindowAverage
sf_df = p.rename(columns={"state_name":"unique_id","date":"ds","y":"y"})[["unique_id","ds","y"]]
train = sf_df[sf_df.ds <= "2023-12-31"]
test  = sf_df[sf_df.ds >= "2024-01-01"]
models = [Naive(), SeasonalNaive(season_length=7), WindowAverage(window_size=7),
          AutoETS(season_length=7), AutoTheta(season_length=7),
          AutoARIMA(season_length=7, max_p=3, max_q=3, max_P=1, max_Q=1, stepwise=True)]
sf = StatsForecast(models=models, freq="D", n_jobs=4)
print("fitting statistical models (rolling 1-step via cross_validation)...", flush=True)
cv = sf.cross_validation(df=train._append(test) if hasattr(train, "_append") else pd.concat([train, test]),
                         h=H, step_size=1, n_windows=min(len(test.ds.unique()), 200), refit=False)
cv = cv.reset_index() if "unique_id" not in cv.columns else cv
for mname in [m.__class__.__name__ if not hasattr(m, "alias") else m.alias for m in models]:
    if mname not in cv.columns: continue
    e = np.abs(cv["y"] - cv[mname])
    res.append(dict(model=mname, family="statistical", MAE=float(e.mean()), n=int(len(e))))
    print(f"  {mname:16s} MAE={e.mean():.3f} [{time.time()-t0:.0f}s]", flush=True)
pd.DataFrame(res).to_csv(os.path.join(OUT, "R6_strong_baselines.csv"), index=False)
print("statistical block done", time.time()-t0)
