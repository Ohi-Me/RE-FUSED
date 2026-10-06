# RE-FUSED-0 — preserved material

The origin of the research, February to August 2026. Full account: [`../stages/stage_00_refused0.md`](../stages/stage_00_refused0.md).

| Folder | What is in it | Why it is here |
|---|---|---|
| `refused_alpha/` | The final pipeline of the stage: the main notebook and its backups, four executed multi-seed copies, result tables and figures, the leakage audit, the publication audit, the related-work notes, and the fix and upgrade modules | This is the pipeline every early result came from |
| `paper_model/` | The market-design model: README, notebooks, the `Next_Model` prototypes and the stored results | The first framing of the settlement question |
| `rl_prototypes/` | RL_V1, RL_V2 and RL_V3 with their logs, plus the RC-CARL manuscript in LaTeX, Word and PDF | The reinforcement-learning line that later became O5 |
| `data_pipeline/` | The cleaning and refining notebooks and their logs | How the early panel was built |
| `REFUSED_Astar_Complete.zip` | An archived A\* prototype (43 MB) | Kept unopened; contents **NEEDS VERIFICATION** if it is ever needed |

## Where to start

1. `refused_alpha/README.md` — what the six contributions N1 to N6 were meant to be.
2. `refused_alpha/LEAKAGE_AUDIT.md` — the audit that triggered the RE-FUSED-1 fix.
3. `refused_alpha/results/tables/` — the stored metrics, including the multi-seed summary.
4. `refused_alpha/notebook/_multiseed/run_all_seeds.log` — evidence of the five-seed run.

## What is not here

- The dataset: one copy is in `../datasets_early/`, and the raw aggregator files are in
  `India_Grid_Study/data/raw/legacy_refused0/`.
- The intermediate cleaning stages `cleaned/` and `cleaned_v2/` (about 1.1 GB): rebuildable with
  `data_pipeline/refine/`.
- Caches: `_ns_cache.pkl` (727 MB) and the harness registries (147 MB), all generated.

## Health warning

The notebooks here were run in an environment that was never pinned, on a GTX 1660 Ti. They are **not**
reproducible in the strict sense. Read the stored outputs; do not expect a re-run to match.
