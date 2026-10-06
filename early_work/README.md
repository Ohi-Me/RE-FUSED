# Early work

Code, results and notes from the stages before the current studies, February to September 2026. They show how the
research got where it is. **They are superseded:** use them to see why the work changed direction, not as evidence
for a claim. Two headline numbers from this material were later shown to be single-seed artefacts, and anything
built on the early market-price column cannot be measured (see below).

What each stage set out to do, what it found and what broke is written up in [`stages/`](stages/).

| Folder | Stage | What it holds |
|---|---|---|
| `stage0_refused0/` | RE-FUSED-0 | Notebooks, audits, result tables and figures, reinforcement-learning prototypes, data-pipeline notebooks |
| `stage1_4_releases/` | RE-FUSED-0.1 to 4 | The five release folders, keeping only what differs between them: reports, notebooks, results, code |
| `stage5_6_refused5/` | RE-FUSED-5 and 6 | Documents, source code, results and audits; the first statement of the forecast-correction question |
| `stage8_laptop_refused8/` | RE-FUSED-8 (laptop) | Status, audit and acquisition logs from before the move to the H100 cluster |

The early datasets are not included: they were assembled partly from non-government aggregator websites, and that
is exactly why the work was later rebuilt from official reports.

## What the early stages found, and why it mattered

| Finding | Value | Why it mattered |
|---|---|---|
| Giving every target its own history as an input | best-model error fell by about half in every model family | The only early accuracy gain, and it came from inputs, not architecture |
| Best early forecaster | gradient-boosted trees on the change from yesterday: scaled error 0.377 against persistence 0.408 | A simple model beat every deep model at that stage |
| Carbon gate | a random-noise control gate beat the real carbon gate (14.08 % against 15.29 % error) | A claimed novelty reversed under a control |
| Regime-coupled dispatch | a 22.5 % volatility reduction was one seed; the same code gave −7.7 % in the next release | Single-seed results were dropped after this |
| Market price in the early panel | 98.2 % of test-period prices were carried forward from an earlier day; 78 % sat at the Rs 10,000 cap | Price skill could not be measured, so all data were rebuilt from official sources |
| The price of adaptivity (RE-FUSED-6) | the most adaptive correction gate was almost never the best | Became the question of the gated-correction study |
