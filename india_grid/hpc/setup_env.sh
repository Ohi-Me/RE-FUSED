#!/bin/bash
# Create the RE-FUSED conda environment on the H100 login node (HPC manual v2.0, section 4: Anaconda, own environment;
# Python 3.10 as recommended in the manual). Usage: bash hpc/setup_env.sh [env_name]
set -eo pipefail
ENV=${1:-refused_h100}
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
    conda create -y -n "$ENV" python=3.10 pip
fi
conda activate "$ENV"
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-h100.txt -c hpc/constraints.txt
# neuralforecast declares torch>=2.9.1; the validated environment runs 3.2.2 on torch 2.5.1, so no dependency upgrade
python -m pip install neuralforecast==3.2.2 --no-deps
python -c "import torch; assert torch.__version__ == '2.5.1+cu121', torch.__version__"
python - <<'PY'
import torch, lightgbm, neuralforecast, stable_baselines3, chronos, pandas
print("python ok; torch", torch.__version__, "CUDA build", torch.version.cuda,
      "| lightgbm", lightgbm.__version__, "| neuralforecast", neuralforecast.__version__,
      "| sb3", stable_baselines3.__version__, "| pandas", pandas.__version__)
PY
echo "ENV_READY $ENV $(date -Iseconds)"
