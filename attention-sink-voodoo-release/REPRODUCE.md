# Reproducing the paper's tables and figures

This document covers reproduction of all tables and figures in the paper. For headline results and quickstart, see [README.md](README.md).

All commands assume:
- You have completed [Setup](README.md#setup) (Python deps, HF_TOKEN, etc.)
- You are in the repo root
- `PYTHONPATH` is set: `export PYTHONPATH="$(pwd)/src:$PYTHONPATH"`

Numbers in parentheses (e.g., "Table 2") refer to the paper's main-text and appendix tables/figures.

## Main paper experiments

### Sink dynamics across layers/timesteps (Figure 2, Table 1)

Step 1 — collect per-step max_incoming_mass / entropy / top-k concentration:

```bash
python experiments/collect_h1_dynamic.py \
    --output_dir results_h1_dynamic \
    --num_samples 8 \
    --num_steps 20
```

Step 2 — render the figure (depends on E3 attribution JSON; see "Text-vs-image attribution" below):

```bash
python figures/make_consolidated_fig.py \
    --h1-csv results_h1_dynamic/h1_dynamic_metrics.csv \
    --e3-json results_e3_multi_gpu/e3_results.json \
    --output figures/fig_consolidated_sink.pdf
```

Convenience wrapper: `bash scripts/plot.sh`. Table 1 statistics are aggregated from `results_h1_dynamic/h1_dynamic_metrics.csv`.

### Dynamic sink intervention main result (Table 2)

```bash
bash scripts/run_geneval_experiments.sh
```

Reproduces conditions A1 (layer 12, top-1), A2 (layers 6/12/18, top-1), and A3 (layer 12, top-5) on the full 553-prompt GenEval set. Output: `results_geneval_A{1,2,3}_*/clip_stats.json` plus per-experiment image directories. Wall-clock: ≈6 hours on 3× A6000.

ImageReward scoring on the produced images:

```bash
python eval/compute_imagereward.py \
    --images_dir results_geneval_A1_layer12_top1 \
    --prompts_file prompts/generation_prompts.txt \
    --output_file results_geneval_A1_layer12_top1/imagereward.json
```

### HPS-v2 evaluation of dynamic sink interventions (Table 3)

```bash
pip install hpsv2
bash scripts/run_hpsv2_eval.sh
```

Output: `*/hps_results.json` next to each input image directory.

### Intervention verification (Table 4)

```bash
python experiments/verify_intervention.py \
    --prompts prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_verification
```

Confirms 44,059× reduction in sink mass under intervention. Also produces the verification appendix figure.

### SDXL cross-architecture (Table 5)

Cross-attention (main paper):

```bash
python experiments/sdxl_sink_experiment.py \
    --prompts_file prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_sdxl_cross \
    --num_prompts 100
```

Self-attention (companion appendix experiment):

```bash
python experiments/sdxl_selfattn_sink_experiment.py \
    --prompts_file prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_sdxl_self \
    --num_prompts 100
```

Both bundled in `scripts/run_supplementary_experiments.sh`.

### Perceptual and distributional effects + sink-vs-random (Tables 6, 7)

> The script behind these tables — `experiments/run_perceptual_delta_delta.py` — originated as an addition during the rebuttal cycle but produces paper-canonical results. It was renamed from `rebuttal_perceptual_delta_delta.py` for clarity.

```bash
pip install lpips
python experiments/run_perceptual_delta_delta.py \
    --prompts_file prompts/generation_prompts.txt \
    --output_dir results_perceptual \
    --num_prompts 64 \
    --target_layer 12 \
    --k_values 1,5 \
    --compute_hps
```

### FID calibration baselines (Tables 9, 18)

Step 1 — generate seed/CFG/steps/scheduler-perturbed image sets:

```bash
python experiments/fid_calibration_experiment.py \
    --prompts_file prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_fid_calibration \
    --num_prompts 100 \
    --model sd3
```

Step 2 — compute pairwise FIDs:

```bash
pip install pytorch_fid
bash scripts/FID_calibration.sh
```

## Appendix experiments

### Score-path dose–response (Tables 10, 11; dose-response curves figure)

```bash
bash scripts/run_full_experiments.sh
python eval/eval_imagereward.py results_full_TIMESTAMP/sweep_score/sweep_sd3 score
```

### Multi-layer intervention (Table 12)

```bash
python experiments/run_multilayer.py \
    --model sd3 \
    --num_samples 32 \
    --prompts prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_multilayer
```

Bundled in `scripts/run_ICML_exps.sh`.

### Phase-specific intervention (Table 13)

```bash
python experiments/run_early_phase.py \
    --model sd3 \
    --num_samples 32 \
    --prompts prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_early_phase
```

Bundled in `scripts/run_ICML_exps.sh`.

### E1 task-type robustness (Table 14)
### E2 sampling sensitivity (Table 15)
### E3 text-vs-image attribution (Table 16)

```bash
bash scripts/run_experiments_4gpu.sh
```

> Note: E1 reads from a previously-completed A1 run (`results_geneval_A1_layer12_top1/`). Run `scripts/run_geneval_experiments.sh` first.

### No-op sanity check (Table 17)

```bash
python experiments/noop_sanity_check.py \
    --prompts prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_noop
```

### CLIP-T budget sweep (Table 19)

```bash
bash scripts/run_k_sweep.sh

python eval/summarize_k_sweep.py \
    --base_dir results_k_sweep \
    --k_values "1,5,10,20,50" \
    --latex_out counterfactual_table.tex
```

### HPS-v2 sink-specificity dose-response (Table 20)

Per-k generation + HPS-v2 scoring:

```bash
for k in 1 10 50; do
    python experiments/hps_v2_k50_validation.py \
        --prompts_file prompts/prompts_geneval_balanced_100.txt \
        --output_dir results_hps_k${k} \
        --top_k ${k} \
        --num_prompts 64
done
```

Sink-specificity (ΔΔ) and dose-response trend:

```bash
python eval/compute_delta_delta.py \
    --images_dir results_hps_k50 \
    --prompts_file prompts/prompts_geneval_balanced_100.txt \
    --num_prompts 64 \
    --output_file delta_delta_k50.json

python eval/compute_dose_trend.py \
    --k10_dir results_hps_k10 \
    --k50_dir results_hps_k50 \
    --prompts_file prompts/prompts_geneval_balanced_100.txt \
    --num_prompts 64 \
    --output_file dose_trend.json
```

### Qualitative drift panels (k=1, k=5)

> Script renamed from `rebuttal_qualitative_panel.py` to `experiments/run_qualitative_panel.py` for clarity. It produces paper-canonical results despite the rebuttal-cycle origin.

```bash
python experiments/run_qualitative_panel.py \
    --p1_dir results_perceptual \
    --output_dir results_qualitative \
    --k 1,5 \
    --n_per_group 5
```

`results_perceptual` is the output directory from `experiments/run_perceptual_delta_delta.py` (Tables 6/7 above).

## Aggregate evaluation helper

`eval/evaluate_all_experiments.py` is a convenience aggregator that re-computes ΔCLIP-T / p-value / LPIPS / FID for a fixed set of experiment directory names (see the `EXPERIMENTS` dict at the top of the file). It depends on prior experiment runs producing the standard `results_geneval_*` directories — run `scripts/run_geneval_experiments.sh` first, or edit the dict to point at your own directories.

```bash
python eval/evaluate_all_experiments.py
```

## Notes

- **Table 8** (Summary of experimental findings) is hand-compiled from results above; no dedicated script.
- **Figure 1** (Conceptual teaser) was manually crafted; not a script-generated figure.
- Several appendix figures (summary heatmap, robustness analysis) were produced by a now-superseded `generate_paper_figures.py` not included in this release. The numerical results behind them are reproducible via the recipes above; for the exact figure renderers, see the git history at the initial release tag.
