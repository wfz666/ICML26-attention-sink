#!/bin/bash
#SBATCH --job-name=geneval-icml-dynamic
#SBATCH --output=slurm-%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:3



: "${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"

set -e

PROMPTS_FILE="prompts/generation_prompts.txt"
mkdir -p logs
NUM_SAMPLES=553
NUM_STEPS=20

echo "============================================================"
echo "GenEval Dynamic Sink Experiments (N=$NUM_SAMPLES)"
echo "============================================================"

# 验证prompts文件
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Error: $PROMPTS_FILE not found!"
    exit 1
fi

PROMPT_COUNT=$(wc -l < "$PROMPTS_FILE")
echo "Found $PROMPT_COUNT prompts in $PROMPTS_FILE"

# 并行运行3个实验
echo ""
echo "Starting A1, A2, A3 in parallel..."

CUDA_VISIBLE_DEVICES=0 python experiments/run_dynamic_sink.py \
    --num_samples $NUM_SAMPLES \
    --num_steps $NUM_STEPS \
    --top_k 1 \
    --layers 12 \
    --prompts $PROMPTS_FILE \
    --output_dir results_geneval_A1_layer12_top1 \
    2>&1 | tee logs/A1_geneval.log &
PID_A1=$!

CUDA_VISIBLE_DEVICES=1 python experiments/run_dynamic_sink.py \
    --num_samples $NUM_SAMPLES \
    --num_steps $NUM_STEPS \
    --top_k 1 \
    --layers 6,12,18 \
    --prompts $PROMPTS_FILE \
    --output_dir results_geneval_A2_multilayer_top1 \
    2>&1 | tee logs/A2_geneval.log &
PID_A2=$!

CUDA_VISIBLE_DEVICES=2 python experiments/run_dynamic_sink.py \
    --num_samples $NUM_SAMPLES \
    --num_steps $NUM_STEPS \
    --top_k 5 \
    --layers 12 \
    --prompts $PROMPTS_FILE \
    --output_dir results_geneval_A3_layer12_top5 \
    2>&1 | tee logs/A3_geneval.log &
PID_A3=$!

# 等待完成
wait $PID_A1 $PID_A2 $PID_A3

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "============================================================"

# 汇总结果
echo ""
echo "Results Summary:"
for dir in results_geneval_A1_layer12_top1 results_geneval_A2_multilayer_top1 results_geneval_A3_layer12_top5; do
    if [ -f "$dir/clip_stats.json" ]; then
        echo ""
        echo "=== $dir ==="
        cat "$dir/clip_stats.json" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Baseline:     {d['baseline_mean']:.4f} ± {d['baseline_std']:.4f}\")
print(f\"  Intervention: {d['intervention_mean']:.4f} ± {d['intervention_std']:.4f}\")
print(f\"  Δ:            {d['delta_mean']:+.4f}\")
print(f\"  95% CI:       [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]\")
print(f\"  p-value:      {d['p_value']:.4f}\")
"
    fi
done