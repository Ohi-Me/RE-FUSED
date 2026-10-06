"""
aggregate_seeds.py -- multi-seed robustness aggregation for RE-FUSED-Alpha.

Parses the executed .ipynb outputs directly (does NOT depend on the on-disk
results/tables/metrics_summary.json, which only ever reflects the single most
recently executed notebook and was overwritten seed-by-seed during the
multiseed sweep). Each seed's own final "Saved: metrics_summary.json" JSON
block, embedded verbatim in that notebook's cell outputs, is the source of
truth for that seed's numbers.

Seeds covered: 42 (main notebook, REFUSED_Alpha_Final.ipynb) + 123, 456, 789,
2024 (notebook/_multiseed/executed_seed*.ipynb).

Outputs:
  results/tables/multiseed_summary.csv   -- long-format per-seed rows
  results/tables/multiseed_summary.json  -- mean/std aggregates + honesty flags
  results/figures/S7_multiseed_robustness.png -- bar chart with error bars

Run standalone:
    python refused_fixes/aggregate_seeds.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nbformat
except ImportError as e:  # pragma: no cover
    raise SystemExit("nbformat is required: pip install nbformat") from e

THESIS_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = THESIS_DIR / "results"
TABLES_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

SEED_NOTEBOOKS = {
    42: THESIS_DIR / "notebook" / "REFUSED_Alpha_Final.ipynb",
    123: THESIS_DIR / "notebook" / "_multiseed" / "executed_seed123.ipynb",
    456: THESIS_DIR / "notebook" / "_multiseed" / "executed_seed456.ipynb",
    789: THESIS_DIR / "notebook" / "_multiseed" / "executed_seed789.ipynb",
    2024: THESIS_DIR / "notebook" / "_multiseed" / "executed_seed2024.ipynb",
}


# --------------------------------------------------------------------------- #
# notebook text extraction
# --------------------------------------------------------------------------- #
def _all_stream_text(nb) -> list[tuple[int, str]]:
    """Return [(cell_index, stream_text), ...] for every code cell with stream output."""
    out = []
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                out.append((i, "".join(o.get("text", ""))))
    return out


def _find_metrics_json(nb) -> dict | None:
    """Locate the cell whose source writes metrics_summary.json and parse the
    JSON blob printed immediately after the 'Saved: ...' banner line."""
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", ""))
        if "Saved: metrics_summary.json" not in src:
            continue
        for o in cell.get("outputs", []):
            if o.get("output_type") != "stream":
                continue
            text = "".join(o.get("text", ""))
            if "Saved: metrics_summary.json" not in text:
                continue
            # JSON body starts at the first '{' after the banner line
            brace = text.find("{")
            if brace == -1:
                continue
            try:
                return json.loads(text[brace:])
            except json.JSONDecodeError:
                # tolerate trailing text after the JSON object by trimming
                # to the matching closing brace via incremental decode
                dec = json.JSONDecoder()
                try:
                    obj, _ = dec.raw_decode(text[brace:])
                    return obj
                except Exception:
                    continue
    return None


_RE_OVERALL_A = re.compile(
    r"OVERALL WINNER\s*:\s*([\w\-]+)\s*\(MASE\(obs\)=([\d.]+)\s*\|\s*WtScore=([\d.]+)"
)
_RE_STAGE_WINNER_B = re.compile(
    r"\*\*\*\s*Stage\s*(\d)\s*Winner:\s*([\w\-]+)\s+MASE\(obs\)=([\d.]+)\s*\[WtScore=([\d.]+)"
)
_RE_CHECKLIST_WINNER = re.compile(r"S2:([\w\-]+)\s+S3:([\w\-]+)\s+S4:([\w\-]+)\s+Winner:([\w\-]+)")


def _find_architecture_winner(nb) -> dict:
    """Extract overall-winner architecture + MASE(obs) via two known print
    formats (older consolidated 'FINAL HEAD-TO-HEAD' cell vs newer per-stage
    '*** Stage N Winner ***' cells). Returns a dict with whatever was found;
    missing fields are None so downstream aggregation can report coverage."""
    result = {
        "stage2_winner": None, "stage2_mase_obs": None, "stage2_wtscore": None,
        "stage3_winner": None, "stage3_mase_obs": None, "stage3_wtscore": None,
        "stage4_winner": None, "stage4_mase_obs": None, "stage4_wtscore": None,
        "overall_winner": None, "overall_mase_obs": None, "overall_wtscore": None,
        "format": None,
    }
    texts = [t for _, t in _all_stream_text(nb)]
    full = "\n".join(texts)

    # Format B: per-stage cells (current main-notebook style)
    stage_hits = _RE_STAGE_WINNER_B.findall(full)
    if stage_hits:
        result["format"] = "per_stage_cells"
        for stage, name, mase_obs, wt in stage_hits:
            key = f"stage{stage}_"
            result[key + "winner"] = name
            result[key + "mase_obs"] = float(mase_obs)
            result[key + "wtscore"] = float(wt)
        # overall = the stage-winner with lowest (best) MASE(obs) among found
        candidates = [
            (result[f"stage{s}_winner"], result[f"stage{s}_mase_obs"], result[f"stage{s}_wtscore"])
            for s in (2, 3, 4)
            if result[f"stage{s}_winner"] is not None
        ]
        # cross-check against explicit checklist "Winner:" token if present
        m = _RE_CHECKLIST_WINNER.search(full)
        if m:
            checklist_winner = m.group(4)
            match = [c for c in candidates if c[0] == checklist_winner]
            if match:
                result["overall_winner"], result["overall_mase_obs"], result["overall_wtscore"] = match[0]
        if result["overall_winner"] is None and candidates:
            best = min(candidates, key=lambda c: c[1])
            result["overall_winner"], result["overall_mase_obs"], result["overall_wtscore"] = best
        return result

    # Format A: single consolidated "FINAL HEAD-TO-HEAD" cell (older seed forks)
    m = _RE_OVERALL_A.search(full)
    if m:
        result["format"] = "consolidated_cell"
        result["overall_winner"] = m.group(1)
        result["overall_mase_obs"] = float(m.group(2))
        result["overall_wtscore"] = float(m.group(3))
        s2 = re.search(r"Stage 2 BiLSTM\s*:\s*([\w\-]+)", full)
        s3 = re.search(r"Stage 3 TFT\s*:\s*([\w\-]+)", full)
        s4 = re.search(r"Stage 4 PatchTST\s*:\s*([\w\-]+)", full)
        if s2:
            result["stage2_winner"] = s2.group(1)
        if s3:
            result["stage3_winner"] = s3.group(1)
        if s4:
            result["stage4_winner"] = s4.group(1)
    return result


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def load_all_seeds() -> pd.DataFrame:
    rows = []
    for seed, path in SEED_NOTEBOOKS.items():
        if not path.exists():
            print(f"  [skip] seed {seed}: notebook not found at {path}")
            continue
        nb = nbformat.read(str(path), as_version=4)
        summary = _find_metrics_json(nb)
        if summary is None:
            print(f"  [warn] seed {seed}: could not locate metrics_summary JSON block")
            continue
        arch = _find_architecture_winner(nb)

        row = {"seed": seed, "notebook": path.name}
        row.update({f"arch_{k}": v for k, v in arch.items()})

        dro = summary.get("dro_calibration", {})
        row["rho_coal"] = dro.get("rho_coal")
        row["rho_recommended"] = dro.get("rho_recommended")
        row["kurtosis_excess"] = dro.get("kurtosis_excess")

        abl = summary.get("mcag_ablation_5way", {})
        for k in ("real", "null", "price", "5min", "regime",
                   "delta_price_vs_real", "delta_5min_vs_real", "delta_regime_vs_real",
                   "n2_confirmed", "n4_supported", "dynamic_gate_superior",
                   "tft_carbon_vs_mcag_real_delta"):
            row[f"abl_{k}"] = abl.get(k)

        rl = summary.get("dispatch_rl", {})
        for arm in ("PPO_standard", "DRO_fixed_lambda", "DRO_vol_scaled_lambda",
                    "DRO_Regime_Adaptive_lambda"):
            d = rl.get(arm, {})
            for metric in ("mean_reward", "std_reward", "cvar90", "worst5pct"):
                row[f"rl_{arm}_{metric}"] = d.get(metric)
        row["rl_std_reduction_dro_pct"] = rl.get("std_reduction_dro_pct")
        row["rl_std_reduction_adp_pct"] = rl.get("std_reduction_adp_pct")

        palmp = summary.get("palmp_gcal", {})
        row["palmp_premium_pct"] = palmp.get("palmp_premium_pct")
        row["mean_carbon_price_inr_per_tco2"] = palmp.get("mean_carbon_price_inr_per_tco2")

        rows.append(row)
        print(f"  [ok] seed {seed}: winner={arch.get('overall_winner')} "
              f"MASE(obs)={arch.get('overall_mase_obs')} n2_confirmed={row.get('abl_n2_confirmed')}")
    return pd.DataFrame(rows)


def _mean_std(df: pd.DataFrame, col: str) -> dict:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std(ddof=1)), 4) if len(s) > 1 else 0.0,
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
    }


def build_aggregate(df: pd.DataFrame) -> dict:
    agg: dict = {"n_seeds": int(len(df)), "seeds": sorted(df["seed"].tolist())}

    # --- architecture winner stability (headline honesty check) ---
    winners = df["arch_overall_winner"].dropna().tolist()
    from collections import Counter
    wc = Counter(winners)
    agg["architecture_winner_stability"] = {
        "counts": dict(wc),
        "n_seeds_reporting": len(winners),
        "modal_winner": wc.most_common(1)[0][0] if wc else None,
        "modal_winner_frac": round(wc.most_common(1)[0][1] / len(winners), 3) if wc else None,
        "stable_across_seeds": len(wc) == 1 if wc else None,
    }
    agg["overall_mase_obs"] = _mean_std(df, "arch_overall_mase_obs")

    # --- N2 ablation (dynamic carbon-aware gating vs null) robustness ---
    n2_vals = df["abl_n2_confirmed"].dropna().tolist()
    n2_true = sum(bool(v) for v in n2_vals)
    agg["n2_ablation_robustness"] = {
        "n_seeds": len(n2_vals),
        "n_confirmed": int(n2_true),
        "frac_confirmed": round(n2_true / len(n2_vals), 3) if n2_vals else None,
        "robust": (n2_true == len(n2_vals)) if n2_vals else None,
    }
    for k in ("real", "null", "price", "5min", "regime"):
        agg[f"ablation_mape_{k}"] = _mean_std(df, f"abl_{k}")
    agg["ablation_delta_price_vs_real"] = _mean_std(df, "abl_delta_price_vs_real")
    agg["ablation_delta_5min_vs_real"] = _mean_std(df, "abl_delta_5min_vs_real")

    # --- DRO calibration stability (should be seed-invariant: sanity check) ---
    agg["rho_coal"] = _mean_std(df, "rho_coal")

    # --- dispatch RL robustness ---
    for arm in ("PPO_standard", "DRO_fixed_lambda", "DRO_vol_scaled_lambda",
                "DRO_Regime_Adaptive_lambda"):
        for metric in ("mean_reward", "std_reward", "cvar90", "worst5pct"):
            col = f"rl_{arm}_{metric}"
            if col in df.columns:
                agg[col] = _mean_std(df, col)
    agg["rl_std_reduction_dro_pct"] = _mean_std(df, "rl_std_reduction_dro_pct")

    return agg


def make_figure(df: pd.DataFrame, agg: dict) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available, skipping figure")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # panel 1: MASE(obs) of the overall winner, per seed
    ax = axes[0]
    seeds = df["seed"].astype(str).tolist()
    mase_vals = pd.to_numeric(df["arch_overall_mase_obs"], errors="coerce")
    colors = ["#2b7a78" if w == agg["architecture_winner_stability"]["modal_winner"] else "#c0392b"
              for w in df["arch_overall_winner"]]
    ax.bar(seeds, mase_vals, color=colors)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="seasonal-naive (MASE=1)")
    ax.set_title("Winner MASE(obs) by seed\n(colour = architecture, see legend text)")
    ax.set_xlabel("seed")
    ax.set_ylabel("MASE (observable target)")
    for x, w, y in zip(seeds, df["arch_overall_winner"], mase_vals):
        if pd.notna(y):
            ax.text(x, y + 0.02, str(w), rotation=90, fontsize=7, ha="center", va="bottom")
    ax.legend(fontsize=7)

    # panel 2: N2 ablation MAPE (real vs null) per seed
    ax = axes[1]
    x = np.arange(len(df))
    w = 0.35
    real_v = pd.to_numeric(df["abl_real"], errors="coerce")
    null_v = pd.to_numeric(df["abl_null"], errors="coerce")
    ax.bar(x - w / 2, real_v, width=w, label="MCAG-Real (dynamic)", color="#2b7a78")
    ax.bar(x + w / 2, null_v, width=w, label="MCAG-Null (ablated)", color="#95a5a6")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("Avg MAPE (%)")
    n2rob = agg["n2_ablation_robustness"]
    ax.set_title(f"N2 ablation: Real vs Null MAPE\nconfirmed in {n2rob['n_confirmed']}/{n2rob['n_seeds']} seeds")
    ax.legend(fontsize=7)

    # panel 3: dispatch RL std_reward across seeds, 3 arms
    ax = axes[2]
    arms = ["PPO_standard", "DRO_fixed_lambda", "DRO_vol_scaled_lambda"]
    means = [agg.get(f"rl_{a}_std_reward", {}).get("mean") for a in arms]
    stds = [agg.get(f"rl_{a}_std_reward", {}).get("std") for a in arms]
    ax.bar(range(len(arms)), means, yerr=stds, capsize=4,
           color=["#7f8c8d", "#2980b9", "#27ae60"])
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(["PPO", "DRO-fixed", "DRO-vol-scaled"], rotation=15)
    ax.set_ylabel("std(reward) mean +/- std across seeds")
    ax.set_title(f"Dispatch RL reward volatility\n(n={agg['n_seeds']} seeds)")

    fig.suptitle("RE-FUSED-Alpha Multi-Seed Robustness (seeds: "
                  f"{', '.join(str(s) for s in agg['seeds'])})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "S7_multiseed_robustness.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def main():
    print("Aggregating multi-seed results from executed notebooks...")
    df = load_all_seeds()
    if df.empty:
        print("No seed data found -- aborting.")
        sys.exit(1)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "multiseed_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved per-seed table: {csv_path}  ({len(df)} seeds)")

    agg = build_aggregate(df)
    json_path = TABLES_DIR / "multiseed_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"Saved aggregate: {json_path}")

    fig_path = make_figure(df, agg)
    if fig_path:
        print(f"Saved figure: {fig_path}")

    print("\n=== HONESTY SUMMARY ===")
    ws = agg["architecture_winner_stability"]
    print(f"Architecture winner stable across {ws['n_seeds_reporting']} seeds: {ws['stable_across_seeds']}"
          f"  (counts={ws['counts']})")
    n2 = agg["n2_ablation_robustness"]
    print(f"N2 (dynamic-gate-beats-null) confirmed in {n2['n_confirmed']}/{n2['n_seeds']} seeds "
          f"(robust={n2['robust']})")
    print(f"Overall MASE(obs): mean={agg['overall_mase_obs']['mean']} "
          f"std={agg['overall_mase_obs']['std']} (n={agg['overall_mase_obs']['n']})")


if __name__ == "__main__":
    main()
