# Attention Sinks in Diffusion Transformers: A Causal Analysis

Code release accompanying the ICML 2026 paper.

📄 [Paper](https://openreview.net/forum?id=QwE8cOtclR) | 🔗 [arXiv](https://arxiv.org/abs/XXXX.XXXXX) | 💻 [Code](https://github.com/wfz666/ICML26-attention-sink)

## Abstract

Attention sinks—tokens that receive disproportionate attention mass—are
assumed to be functionally important in autoregressive language models, but
their role in diffusion transformers remains unclear.
We present a causal analysis in text-to-image diffusion, dynamically identifying
dominant attention recipients per timestep and suppressing them via paired,
training-free interventions on the score and value paths.
Across 553 GenEval prompts on Stable Diffusion 3 (with SDXL corroboration),
removing these sinks does not degrade text–image alignment (CLIP-T) or
preference proxies (ImageReward, HPS-v2) at *k*=1; only under stronger
interventions (*k*≥10) does HPS-v2 exhibit a metric-dependent boundary,
while CLIP-T remains robust throughout.
The perceptual shifts induced by suppression are nonetheless
*sink-specific*—∼6× larger than equal-budget random
masking—revealing an empirical dissociation between trajectory-level
perturbation and *semantic alignment* in diffusion transformers.

## Setup

### Python and dependencies

Tested on **Python 3.10**. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A handful of evaluation scripts use third-party metrics packages that are
heavyweight or version-sensitive; install them only as needed (commented out
at the bottom of `requirements.txt`):

| Package        | Used by |
|----------------|---------|
| `hpsv2`        | `eval/run_hpsv2_eval.py`, `experiments/hps_v2_k50_validation.py` |
| `image-reward` | `eval/compute_imagereward.py`, `eval/eval_imagereward.py` |
| `lpips`        | `experiments/run_perceptual_delta_delta.py` |
| `pytorch_fid`  | `scripts/FID_calibration.sh` |

### GPU and runtime

Tested on **4× NVIDIA A6000 (48 GB each)**. A single A6000 is sufficient for
the Quickstart and most inference-only experiments; multi-GPU is recommended
for full GenEval reproduction.

### Models and access

Stable Diffusion 3 and SDXL are loaded via the `diffusers` library and pulled
from the Hugging Face Hub on first use. SD3 is gated, so you must accept the
license at <https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers>
and export a token before running anything:

```bash
export HF_TOKEN=hf_xxx           # your personal token
huggingface-cli login            # alternative
```

Every shell driver in `scripts/` checks `HF_TOKEN` is set and aborts
otherwise. The release does not ship any token.

### Prompts

Two prompt files are included:

- `prompts/generation_prompts.txt` — 553 GenEval prompts (paper main result).
- `prompts/prompts_geneval_balanced_100.txt` — 100-prompt balanced subset used
  for the *k*-sweep, HPS-v2 validation, and FID calibration experiments.

## Repository structure

```
attention-sink-voodoo/
├── README.md
├── LICENSE                    # MIT
├── .gitignore
├── requirements.txt
├── prompts/                   # GenEval prompt files
├── src/                       # Importable library modules (added to PYTHONPATH)
│   ├── dynamic_sink_processor.py    # SD3 joint-attn dynamic sink processor + patcher
│   ├── sink_analysis.py             # v3 framework for H1/H2/sweep experiments
│   ├── quality_metrics.py           # Paired-CLIP / sweep evaluation
│   └── hpsv2_evaluator.py           # HPS-v2 evaluator helper module
├── experiments/               # Generation drivers
├── eval/                      # Scoring drivers (no image generation)
├── figures/                   # Figure renderers
├── scripts/                   # Bash drivers; each sets PYTHONPATH and calls into
│                              #   experiments/, eval/, figures/ from the repo root
└── tests/                     # Smoke tests (test_processor.py)
```

## Quickstart (≈30 minutes, single A6000)

A small smoke test that exercises the main intervention pipeline on 32
prompts at the paper's `top_k=1, layer=12` configuration:

```bash
export HF_TOKEN=hf_xxx
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

python experiments/run_dynamic_sink.py \
    --num_samples 32 \
    --num_steps 20 \
    --top_k 1 \
    --layers 12 \
    --prompts prompts/generation_prompts.txt \
    --output_dir results_quickstart
```

This generates baseline + intervention images for the first 32 prompts,
reports paired-CLIP statistics, and writes `results_quickstart/clip_stats.json`.

> **Note on flags.** `run_dynamic_sink.py` exposes
> `--num_samples` (not `--num_prompts`) to limit prompt count, and
> `--prompts` (not `--prompts_file`). A few sibling scripts use the longer
> `--prompts_file` / `--num_prompts` form — always check the script's
> argparse if in doubt.

## Main results

The paper's headline findings can be reproduced as:

| Finding | Tables | Command |
|---|---|---|
| Non-necessity for alignment | Table 2 | `bash scripts/run_geneval_experiments.sh` |
| SDXL cross-architecture | Table 5 | `python experiments/sdxl_sink_experiment.py ...` |
| Sink-specific perceptual dissociation (~6×) | Tables 6, 7 | `python experiments/run_perceptual_delta_delta.py ...` |
| Metric-dependent boundary at k≥10 | Tables 19, 20 | `bash scripts/run_k_sweep.sh` |

For appendix experiments and full table-by-table recipes, see "Full reproduction" below.

## Full reproduction of paper experiments

The headline result (Table 2 — Dynamic sink intervention, *N*=553) is
reproduced by:

```bash
export HF_TOKEN=hf_xxx
bash scripts/run_geneval_experiments.sh
```

Wall-clock: ≈6 hours on 3× A6000 (the script parallelises A1/A2/A3 across
three GPUs).

The remaining experiments map to the following table/figure references in the
camera-ready paper. **Numbering follows the order tables appear in the
camera-ready PDF** (the paper's introductory text uses descriptive names, not
table numbers).

### Figure 2 — Sink dynamics across layers and timesteps (`fig:h1_dynamics`)

```bash
# Step 1: collect per-step max_incoming_mass / entropy / top-k concentration
python experiments/collect_h1_dynamic.py \
    --output_dir results_h1_dynamic \
    --num_samples 8 \
    --num_steps 20

# Step 2: render the figure (depends on E3 attribution JSON; see Table 16 below)
python figures/make_consolidated_fig.py \
    --h1-csv results_h1_dynamic/h1_dynamic_metrics.csv \
    --e3-json results_e3_multi_gpu/e3_results.json \
    --output figures/fig_consolidated_sink.pdf
```

Convenience wrapper: `bash scripts/plot.sh`.

### Table 1 — Attention concentration statistics (`tab:h1_stats`)

Aggregated from `results_h1_dynamic/h1_dynamic_metrics.csv` produced by
`experiments/collect_h1_dynamic.py`.

### Table 2 — Dynamic sink intervention main result (`tab:dynamic_sink`)

```bash
bash scripts/run_geneval_experiments.sh
```

Reproduces conditions A1 (layer 12, top-1), A2 (layers 6,12,18, top-1),
and A3 (layer 12, top-5) on the full 553-prompt GenEval set. Output:
`results_geneval_A{1,2,3}_*/clip_stats.json` plus per-experiment image
directories. ImageReward scoring is then computed via:

```bash
python eval/compute_imagereward.py \
    --images_dir results_geneval_A1_layer12_top1 \
    --prompts_file prompts/generation_prompts.txt \
    --output_file results_geneval_A1_layer12_top1/imagereward.json
```

### Table 3 — HPS-v2 evaluation of dynamic sink interventions (`tab:hpsv2`)

```bash
pip install hpsv2
bash scripts/run_hpsv2_eval.sh
```

Output: `*/hps_results.json` next to each input image directory.

### Table 4 — Dynamic sink intervention verification (`tab:dynamic_verification`)

```bash
python experiments/verify_intervention.py \
    --prompts prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_verification
```

Also produces `fig:verification_appendix` (44,059× sink-mass reduction).

### Table 5 — SDXL cross-architecture validation (`tab:sdxl`)

```bash
# Cross-attention (main paper)
python experiments/sdxl_sink_experiment.py \
    --prompts_file prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_sdxl_cross \
    --num_prompts 100

# Self-attention (companion appendix experiment)
python experiments/sdxl_selfattn_sink_experiment.py \
    --prompts_file prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_sdxl_self \
    --num_prompts 100
```

Both runs are also bundled in `scripts/run_supplementary_experiments.sh`.

### Table 6 — Perceptual and distributional effects (`tab:perceptual`)
### Table 7 — Sink vs. random masking at equal budget (`tab:sink_vs_random_lpips`)

> The script behind these two tables is `experiments/run_perceptual_delta_delta.py`,
> which originated as an addition during the rebuttal cycle but produces
> paper-canonical results (Tables 6/7 and Figures `fig:qualitative_k1` /
> `fig:qualitative_k5`). It was renamed from `rebuttal_perceptual_delta_delta.py`
> for clarity.

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

### Table 9 — FID calibration baselines, short table (`tab:fid_context`)
### Table 18 — FID calibration baselines, full N=100 (`tab:fid_calibration`)

```bash
# Step 1: generate seed/CFG/steps/scheduler-perturbed image sets
python experiments/fid_calibration_experiment.py \
    --prompts_file prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_fid_calibration \
    --num_prompts 100 \
    --model sd3

# Step 2: compute pairwise FIDs
pip install pytorch_fid
bash scripts/FID_calibration.sh
```

### Figure (Appendix) — Dose–response curves (`fig:dose_response_appendix`)
### Table 10 — Score-path dose–response, CLIP-T (`tab:score_sweep_clip_appendix`)
### Table 11 — Score-path dose–response, ImageReward (`tab:score_sweep_ir_appendix`)

```bash
bash scripts/run_full_experiments.sh
# After it finishes, ImageReward on the score sweep:
python eval/eval_imagereward.py results_full_TIMESTAMP/sweep_score/sweep_sd3 score
```

### Figure (Appendix) — Intervention verification (`fig:verification_appendix`)

Produced by `experiments/verify_intervention.py` (see Table 4 above).

### Table 12 — Multi-layer intervention (`tab:multilayer_appendix`)

```bash
python experiments/run_multilayer.py \
    --model sd3 \
    --num_samples 32 \
    --prompts prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_multilayer
```

Bundled in `scripts/run_ICML_exps.sh`.

### Table 13 — Phase-specific intervention (`tab:phase_appendix`)

```bash
python experiments/run_early_phase.py \
    --model sd3 \
    --num_samples 32 \
    --prompts prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_early_phase
```

Bundled in `scripts/run_ICML_exps.sh`.

### Table 14 — E1 task-type robustness (`tab:e1_appendix`)
### Table 15 — E2 sampling sensitivity (`tab:e2_appendix`)
### Table 16 — E3 text-vs-image attribution (`tab:e3_appendix`)

```bash
bash scripts/run_experiments_4gpu.sh
```

Note: E1 reads from a previously-completed A1 run
(`results_geneval_A1_layer12_top1/`). Run `scripts/run_geneval_experiments.sh`
first.

### Table 17 — No-op sanity check (`tab:sanity_check`)

```bash
python experiments/noop_sanity_check.py \
    --prompts prompts/prompts_geneval_balanced_100.txt \
    --output_dir results_noop
```

### Table 19 — CLIP-T under varying masking budgets (`tab:clip_k_sweep`)

```bash
bash scripts/run_k_sweep.sh
python eval/summarize_k_sweep.py \
    --base_dir results_k_sweep \
    --k_values "1,5,10,20,50" \
    --latex_out counterfactual_table.tex
```

### Table 20 — Sink-specificity under HPS-v2 (`tab:sink_specificity_appendix`)

```bash
# Per-k generation + HPS-v2 scoring
for k in 1 10 50; do
    python experiments/hps_v2_k50_validation.py \
        --prompts_file prompts/prompts_geneval_balanced_100.txt \
        --output_dir results_hps_k${k} \
        --top_k ${k} \
        --num_prompts 64
done

# Sink-specificity (ΔΔ) and dose-response trend
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

### Figure (Appendix) — Qualitative drift panels at k=1 and k=5 (`fig:qualitative_k1`, `fig:qualitative_k5`)

> Script renamed from `rebuttal_qualitative_panel.py` to
> `experiments/run_qualitative_panel.py` for clarity. It produces
> paper-canonical results despite the rebuttal-cycle origin.

```bash
python experiments/run_qualitative_panel.py \
    --p1_dir results_perceptual \
    --output_dir results_qualitative \
    --k 1,5 \
    --n_per_group 5
```

`results_perceptual` is the output directory from
`experiments/run_perceptual_delta_delta.py` (Tables 6/7 above).

## Aggregate evaluation helper

`eval/evaluate_all_experiments.py` is a convenience aggregator that re-computes
ΔCLIP-T / p-value / LPIPS / FID for a fixed set of experiment directory names
(see the `EXPERIMENTS` dict at the top of the file). It depends on prior
experiment runs producing the standard `results_geneval_*` directories — run
`scripts/run_geneval_experiments.sh` first, or edit the dict to point at your
own directories.

```bash
python eval/evaluate_all_experiments.py
```

## Smoke test

A small no-GPU-dependency smoke test verifies that the dynamic-sink processor
patches and unpatches cleanly and that no-op produces pixel-identical output:

```bash
python tests/test_processor.py
```

## Citation

```bibtex
@inproceedings{wu2026attention,
  title     = {Attention Sinks in Diffusion Transformers: A Causal Analysis},
  author    = {Wu, Fangzheng and Summa, Brian},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

The proceedings volume / pages will be added once PMLR publishes the camera
ready.

## License

This release is distributed under the [MIT License](LICENSE).

## Acknowledgements

This work was supported by DOE ASCR (Award DE-SC0022873), the National
Institutes of Health (Award R01GM143789), and the Advanced Research Projects
Agency for Health (ARPA-H, Award D24AC00338-00). The content is solely the
responsibility of the authors and does not necessarily represent the official
views of the funding agencies.

## Contact

- Fangzheng Wu (corresponding author) — <fwu66666666@gmail.com>
- Brian Summa — <bsumma@tulane.edu>

For bug reports and questions about reproducing results, please open an issue
on [GitHub](https://github.com/wfz666/ICML26-attention-sink/issues).
