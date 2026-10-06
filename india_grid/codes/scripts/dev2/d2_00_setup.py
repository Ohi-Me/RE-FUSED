"""dev2 setup (runs inside a GPU job, never on the login node): add XGBoost to the conda env and fetch the two
pretrained Chronos models into data/hf_cache. Writes what it did to results/dev2/setup.json.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused.paths import DATA, RES  # noqa: E402

info = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}

# XGBoost wheels on PyPI already carry CUDA support; keep numpy and torch as they are
r = subprocess.run([sys.executable, "-m", "pip", "install", "xgboost>=2.1,<4"], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
import importlib  # noqa: E402
importlib.invalidate_caches()
try:
    import xgboost
    info["xgboost"] = xgboost.__version__
except ImportError as e:
    info["xgboost"] = f"failed: {e}"

os.environ["HF_HOME"] = os.path.join(DATA, "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "0"
from huggingface_hub import snapshot_download  # noqa: E402

for repo in ("amazon/chronos-2", "amazon/chronos-bolt-base"):
    try:
        path = snapshot_download(repo)
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)
        info[repo] = dict(path=path, bytes=size)
    except Exception as e:  # keep going; the model is then marked not available
        info[repo] = f"failed: {e}"
    print(repo, info[repo], flush=True)

import numpy, torch  # noqa: E401,E402
info.update(numpy=numpy.__version__, torch=torch.__version__, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
os.makedirs(os.path.join(RES, "dev2"), exist_ok=True)
json.dump(info, open(os.path.join(RES, "dev2", "setup.json"), "w"), indent=1)
print(json.dumps(info, indent=1))
if str(info["xgboost"]).startswith("failed"):
    sys.exit(1)
