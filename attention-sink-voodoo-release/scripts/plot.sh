#!/bin/bash
# Render the consolidated Figure 2 (fig:h1_dynamics).
#
# Step 1 collects per-step max_incoming_mass / entropy / top-k concentration
# into results_h1_dynamic/h1_dynamic_metrics.csv.
# Step 2 renders the figure, combining (a) the H1 dynamics with (b) the
# text-vs-image attribution stack from a previously-completed E3 run
# (see scripts/run_experiments_4gpu.sh).

set -e

: "${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"

cd "$(dirname "$0")/.."

# Step 1: collect H1 dynamics CSV (~10 min on one GPU)
python experiments/collect_h1_dynamic.py \
    --output_dir results_h1_dynamic \
    --num_samples 8 \
    --num_steps 20

# Step 2: render Figure 2
# E3 attribution JSON is produced by scripts/run_experiments_4gpu.sh.
python figures/make_consolidated_fig.py \
    --h1-csv results_h1_dynamic/h1_dynamic_metrics.csv \
    --e3-json results_e3_multi_gpu/e3_results.json \
    --output figures/fig_consolidated_sink.pdf
