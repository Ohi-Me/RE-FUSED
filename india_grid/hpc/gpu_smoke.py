"""Short GPU check on one MIG instance: CUDA, matrix product, a 30-step PatchTST fit (neuralforecast) and a BiLSTM step."""
import time

import numpy as np
import pandas as pd
import torch

print("torch", torch.__version__, "CUDA", torch.version.cuda, "available", torch.cuda.is_available(),
      "devices", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)
a = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize(); t0 = time.time(); (a @ a).sum().item(); torch.cuda.synchronize()
print(f"matmul 4096: {time.time() - t0:.3f}s; free/total GB", [round(x / 1e9, 1) for x in torch.cuda.mem_get_info()], flush=True)
lstm = torch.nn.LSTM(40, 64, 2, batch_first=True, bidirectional=True).cuda()
out, _ = lstm(torch.randn(512, 14, 40, device="cuda")); out.mean().backward()
print("bilstm step ok", tuple(out.shape), flush=True)
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import PatchTST
ds = pd.date_range("2020-01-01", periods=400)
df = pd.concat([pd.DataFrame({"unique_id": f"s{i}", "ds": ds, "y": np.sin(np.arange(400) / 7 + i) + 0.1 * np.random.randn(400)})
                for i in range(5)])
m = PatchTST(h=4, input_size=14, patch_len=7, stride=3, hidden_size=32, n_heads=4, loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
             max_steps=30, accelerator="gpu", devices=1, enable_progress_bar=False, logger=False, enable_checkpointing=False)
t0 = time.time()
cv = NeuralForecast(models=[m], freq="D").cross_validation(df=df, n_windows=3, step_size=1, refit=False)
print(f"patchtst gpu fit+cv ok: {len(cv)} rows, {time.time() - t0:.1f}s", flush=True)
print("GPU_SMOKE_OK", flush=True)
