#!/bin/bash
#SBATCH --job-name=k_sweep
#SBATCH --output=slurm-%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4



: "${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"
# Counterfactual Ablation k-Sweep Experiment
# 
# 实验设计：
# - k ∈ {1, 5, 10, 20, 50}
# - N = 64 prompts (足够统计显著性，同时节省时间)
# - Layer 12
# - 4 GPU 并行
#
# 预期结果：
# 1. 如果 k=5/10 仍然所有 mode CI∋0 → "模型对稀疏 ablation 普遍鲁棒"
# 2. 如果 k 增大后 random/bottom CI∌0 但 top_sink CI∋0 → "sinks specifically non-necessary"
# 3. 如果 high_outgoing_query CI∌0 → "anti-sink control 有效，ablation 不是无效"

set -e

PROMPTS_FILE=${1:-"prompts/generation_prompts.txt"}
BASE_OUTPUT_DIR=${2:-"results_k_sweep"}
NUM_GPUS=${3:-4}
NUM_PROMPTS=${4:-64}

echo "=================================="
echo "Counterfactual k-Sweep Experiment"
echo "=================================="
echo "Prompts file: $PROMPTS_FILE"
echo "Output base dir: $BASE_OUTPUT_DIR"
echo "Num GPUs: $NUM_GPUS"
echo "Num prompts: $NUM_PROMPTS"
echo ""

# k values to sweep
K_VALUES=(1 5 10 20 50)

for k in "${K_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running k=$k"
    echo "=========================================="
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/k${k}"
    
    python experiments/ablation_counterfactual_v3.py \
        --prompts_file "$PROMPTS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --num_prompts "$NUM_PROMPTS" \
        --target_layer 12 \
        --top_k "$k" \
        --num_gpus "$NUM_GPUS"
    
    echo "k=$k done. Results saved to $OUTPUT_DIR"
done

echo ""
echo "=========================================="
echo "All k-sweep experiments complete!"
echo "=========================================="
echo ""
echo "Results summary:"
for k in "${K_VALUES[@]}"; do
    echo ""
    echo "--- k=$k ---"
    cat "${BASE_OUTPUT_DIR}/k${k}/counterfactual_results.json" 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"N={data['n_prompts']}, Layer={data['target_layer']}, k={data['top_k']}\")
for mode, res in data['modes'].items():
    ci_str = f\"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]\"
    zero_str = '✓' if res['ci_includes_zero'] else '✗'
    print(f\"  {mode:<22} Δ={res['delta_mean']:+.4f} CI={ci_str} CI∋0?{zero_str}\")
" || echo "  (results not found)"
done
