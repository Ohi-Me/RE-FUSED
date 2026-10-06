# Risk-Constrained Carbon-Aware RL — A* Research Project
## Complete Setup & Run Guide

---

## 📁 Folder Structure (Create This Exactly)

```
final_RL/
├── data/
│   ├── RL_TRAIN_DATA.csv              ← your existing train data
│   └── RL_TEST_DATA.csv               ← your existing test data
│
├── notebooks/
│   └── carbon_aware_rl_astar_complete.ipynb   ← NEW MAIN NOTEBOOK
│
├── results/
│   ├── models/
│   │   ├── carbon_twin_best.pt        ← auto-saved: LSTM carbon twin
│   │   ├── qf_demand.pt               ← auto-saved: demand quantile model
│   │   ├── qf_renew.pt                ← auto-saved: renewable quantile model
│   │   ├── ppo_lagrangian_final.zip   ← auto-saved: trained RL agent
│   │   ├── rl_scaler.pkl              ← auto-saved: RL state scaler
│   │   └── config.json                ← auto-saved: config
│   │
│   ├── figures/
│   │   ├── 01_carbon_twin.png         ← Twin loss + pred vs true + MCI
│   │   ├── 02_quantile_forecast.png   ← 90% PI demand forecast
│   │   ├── 03_lagrangian_training.png ← λ convergence + reward curves
│   │   ├── 04_market_pricing.png      ← 3-scheme market comparison
│   │   ├── 05_dual_temporal_validation.png ← expanding + sliding window
│   │   └── 06_ablation_study.png      ← 5-component ablation bars
│   │
│   └── logs/
│       ├── final_results.json         ← complete metrics for paper
│       ├── expanding_window.csv
│       ├── sliding_window.csv
│       ├── ablation.csv
│       └── market_simulation.csv
│
└── README_ASTAR.md                    ← this file
```

---

## 🔧 Installation (Run Once)

```bash
pip install gymnasium==0.29.1
pip install stable-baselines3==2.2.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib tqdm
```

---

## 🚀 How to Run (Full Pipeline ~30–45 min on GPU, ~2h on CPU)

Open `notebooks/carbon_aware_rl_astar_complete.ipynb` and run cells top to bottom. Each section is clearly marked.

| Section | What It Does | Runtime |
|---------|-------------|---------|
| §2 Data | Load + feature engineering | < 1 min |
| §3 Carbon Twin | Train LSTM emission model (60 epochs) | ~5 min GPU |
| §4 Quantile Forecaster | Train demand + renewable PI models | ~4 min GPU |
| §5 Embeddings | Pre-compute quantile states for RL | ~1 min |
| §6 Environment | Build GridDispatchEnvV2 (5D actions) | < 1 min |
| §7 PPO Training | Lagrangian Constrained PPO (150k steps) | ~15 min GPU |
| §8 Market | Carbon-adjusted merit order simulation | ~2 min |
| §9 Evaluation | Dual temporal validation + stats tests | ~5 min |
| §10 Ablation | 5-component ablation study | ~5 min |

---

## 📊 What v2 Adds vs v1 (Verdict Gaps Fixed)

### ❌ → ✅ Component 1: Carbon Digital Twin
- **v1**: `emission_proxy = 1 - renewable_ratio` (heuristic)
- **v2**: LSTM model trained on `[G_t, D_t, CS_t, O_t, season, history]`
  - Heteroscedastic loss (uncertainty-aware)
  - Autograd-computed Marginal Carbon Intensity (MCI)
  - Distributional output: mean + log-variance

### ❌ → ✅ Component 2: Probabilistic Forecasting
- **v1**: No uncertainty modeling, RL sees only point estimates
- **v2**: Separate LSTM quantile forecasters for demand AND renewables
  - Pinball loss: `L = Σ_τ max(τ(y-ŷ), (τ-1)(y-ŷ))`
  - Outputs: q05, q50, q95
  - 90% PI coverage reported and calibration verified
  - 6 quantile values injected into RL observation state

### ❌ → ✅ Component 3: Multi-Source Dispatch Action
- **v1**: `action ∈ [-1, 1]` scalar adjusting renewable by ±10%
- **v2**: `action ∈ [0,1]^5` = (coal, gas, hydro, solar, wind) fractions
  - Softmax-normalised inside `step()`
  - Emission computed from dispatch mix using per-source emission factors

### ❌ → ✅ Component 4: Lagrangian Constrained PPO
- **v1**: Standard PPO with reward penalty
- **v2**: True dual-variable Lagrangian update
  - `LagrangianCallback` updates λ every rollout:
    `λ_{k+1} = max(0, λ_k + η * violation_k)`
  - Two constraints: shortage fraction ≤ 0.02, emission ≤ 0.70
  - λ values injected back into environment at each rollout

### ❌ → ✅ Component 5: Market Pricing Layer
- **v1**: No market simulation
- **v2**: Full `CarbonAwareMarket` class
  - Generator fleet: 7 units (coal×2, gas×2, hydro, solar, wind)
  - Three schemes: Historical | Carbon-Only | Hybrid (proposed)
  - Metrics: MCP, emission intensity, generator welfare, price volatility
  - Statistical comparison across full test set

---

## 📝 Paper Section → Code Mapping

| Paper Claim | Implementation |
|-------------|---------------|
| "Dynamic emission function f_θ" | `CarbonDigitalTwin` LSTM |
| "Marginal Carbon Intensity" | `compute_mci()` via autograd |
| "TFT quantile outputs" | `QuantileForecaster` + pinball loss |
| "State s_t = (D^dist, R^dist, ...)" | 6 quantile cols in obs + twin_emission |
| "Action a_t = (g_coal, g_gas, ...)" | 5D Box action space |
| "CVaR_α(E)" | Rolling quantile in `step()` |
| "λ_{k+1} = λ_k + η·violation" | `LagrangianCallback._on_rollout_end()` |
| "Score_i = b_i + γ·ε_i" | `CarbonAwareMarket._clear_market()` |
| "P_i = MCP - δ·ε_i" | Hybrid scheme payment calculation |
| "Expanding + Sliding window" | Sections 9.1 and 9.2 |
| "Wilcoxon test" | Sliding window stability test |
| "Ablation study" | Section 10 (5 variants) |

---

## 📈 Key Metrics to Report in Paper

From `results/logs/final_results.json`:

```json
{
  "carbon_twin": { "test_rmse": ..., "test_mae": ..., "mci_mean": ... },
  "quantile_forecaster": { "demand_coverage": ..., "renew_coverage": ... },
  "lagrangian_ppo": {
    "final_lambda_shortage": ...,
    "final_lambda_emission": ...,
    "test": { "mean_emission": ..., "cvar_emission": ..., "mean_shortage": ... }
  },
  "market_comparison": {
    "hybrid": { "avg_emission": ... },
    "historical": { "avg_emission": ... }
  },
  "ablation": { "Full Model": ..., "No CVaR": ..., ... }
}
```

---

## ⚙️ Tuning Tips

- Increase `total_timesteps` to 300k+ for stronger convergence
- Increase `twin_epochs` to 100 for better digital twin
- Adjust `shortage_budget` (lower = stricter reliability constraint)
- Try `policy_kwargs = {'net_arch': [512, 512, 256]}` for more capacity
- Use `lambda_lr = 0.005` if dual variables oscillate
