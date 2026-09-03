# SC-PCP workspace guide

## Environment

- Work from `/home/ubuntu/zmh/sc-pcp`.
- Use the `ucp` conda environment for Python commands.
- GPU experiments default to `cuda:0,cuda:1`; do not silently fall back to CPU.
- Generated artifacts live under `results/` and are intentionally not source-controlled.

## Canonical method and metrics

- There is one paper method named `SC-PCP`: uncapped committed-prefix importance-weighted marginal calibration with free stagewise radii.
- Its implementation is `src/scpcp/marginal_prefix.py`; integration is in `src/scpcp/experiment.py`.
- The canonical comparison names are exactly `Standard CP`, `ACI`, `MFCS`, `SPCI`, `PRC`, and `SC-PCP`.
- The primary coverage metric is `min_t mean_seed(C_seed,t)`. Do not replace it with `mean_seed(min_t C_seed,t)` or MeanCov.
- The claim is asymptotic per-step marginal coverage. Do not describe it as finite-sample, distribution-free, PAC, or data-conditional coverage.
- Historical COT, profile, LCB, and ordered-IUT code is diagnostic only and must not generate the paper's `SC-PCP` row.

## Main commands

Run the complete two-GPU paper suite:

```bash
conda run -n ucp python scripts/run_paper_suite.py --sections rq1,rq3 --datasets synthetic,mimic_iv,mimic_cxr,eicu,inspire --devices cuda:0,cuda:1 --output-root results/work/paper_final
```

Resume the exact same suite by appending `--resume`. Do not change its source, config, datasets, or seed set in place.

Render PDF-only paper outputs:

```bash
conda run -n ucp python tools/render_paper_results.py --input results/work/paper_final --output results/paper_final
```

Run validation:

```bash
conda run -n ucp pytest -q tests/per_step
```

## Editing rules

- Preserve user changes and unrelated dirty-worktree files.
- Use `rg`/`rg --files` for search and `apply_patch` for source edits.
- Keep primary records to the six canonical methods; place oracle/decomposition studies in separate diagnostic artifacts.
- A fresh run must use an empty output root. Resume must fail closed on malformed or provenance-mismatched artifacts.
- Final paper output directories contain PDF files only; figures use Times New Roman and the paper renderer.

## Authoritative documentation

- `README.md`: runnable overview and entry points.
- `docs/final_method.md`: full method definition and claim boundary.
- `docs/evaluation_metrics.md`: exact metric formulas and uncertainty intervals.
- `docs/baselines_and_settings.md`: baseline adapters and information budgets.
- `docs/revised_experiment_plan_zh.md`: final RQ-oriented experiment protocol.
- `docs/main_results_20260822.md`: frozen complete-suite results and honest interpretation.
- `docs/formal_experiments_20260825.md`: latest exact-MDP, controlled all-six, and copula-gate formal results; older controlled artifacts remain protocol-specific history.
- `docs/formal_experiments_20260826.md`: post-freeze horizon/overlap, calibration-size, propensity, strict-split, and gated dataset-native controlled-clinical evidence; this is the latest formal evidence record.
- `docs/figure_portfolio_20260826.md`: submission figure/table order, source bundles, captions, and claim boundaries.
