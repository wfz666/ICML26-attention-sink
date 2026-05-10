#!/bin/bash
#SBATCH --job-name=ICML-exps
#SBATCH --output=slurm-%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:3



: "${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"

#
# ICML补强实验 - 3 GPU并行
# ===========================
# GPU 0: 多层干预 (SD3, ~1小时)
# GPU 1: SDXL跨架构 (~2小时)
# GPU 2: Early-phase (SD3, ~2小时)
#
# Usage:
#   chmod +x run_icml_experiments.sh
#   ./run_icml_experiments.sh
#

set -e

# 配置
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="./results_icml_${TIMESTAMP}"
NUM_SAMPLES=32
PROMPTS_FILE="prompts/prompts_geneval_balanced_100.txt"

mkdir -p $OUTPUT_BASE

echo "============================================================"
echo "ICML Supplementary Experiments - 3 GPU Parallel"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "Output: $OUTPUT_BASE"
echo "Samples: $NUM_SAMPLES"
echo "Prompts: $PROMPTS_FILE"
echo "============================================================"

# 检查prompts文件
if [ -f "$PROMPTS_FILE" ]; then
    PROMPTS_ARG="--prompts $PROMPTS_FILE"
    echo "✓ Using prompts from: $PROMPTS_FILE"
else
    PROMPTS_ARG=""
    echo "⚠ WARNING: $PROMPTS_FILE not found, using default prompts"
fi

echo ""
echo "Starting experiments at $(date)"
echo ""

# -----------------------------------------------------------------------------
# GPU 0: 多层干预 (6+12+18同时)
# -----------------------------------------------------------------------------
echo "[GPU 0] Starting Multilayer Intervention (SD3)..."
CUDA_VISIBLE_DEVICES=0 python experiments/run_multilayer.py \
    --model sd3 \
    --num_samples $NUM_SAMPLES \
    $PROMPTS_ARG \
    --output_dir $OUTPUT_BASE/multilayer \
    2>&1 | tee $OUTPUT_BASE/log_multilayer.txt &
PID0=$!
echo "  PID: $PID0"

# -----------------------------------------------------------------------------
# GPU 1: SDXL跨架构验证
# -----------------------------------------------------------------------------
echo "[GPU 1] Starting SDXL Cross-Architecture Validation..."
# NOTE: The original script run_sdxl_ablation.py was superseded by the
# paper-implementation files experiments/sdxl_sink_experiment.py
# (cross-attn) and experiments/sdxl_selfattn_sink_experiment.py (self-attn).
# Below we use the self-attn version, which matches the original ablation
# scope. See scripts/run_supplementary_experiments.sh for the cross-attn run.
CUDA_VISIBLE_DEVICES=1 python experiments/sdxl_selfattn_sink_experiment.py \
    --prompts_file "$PROMPTS_FILE" \
    --num_prompts $NUM_SAMPLES \
    --output_dir $OUTPUT_BASE/sdxl_ablation \
    --device cuda \
    2>&1 | tee $OUTPUT_BASE/log_sdxl.txt &
PID1=$!
echo "  PID: $PID1"

# -----------------------------------------------------------------------------
# GPU 2: Early-phase干预
# -----------------------------------------------------------------------------
echo "[GPU 2] Starting Early-Phase Intervention (SD3)..."
CUDA_VISIBLE_DEVICES=2 python experiments/run_early_phase.py \
    --model sd3 \
    --num_samples $NUM_SAMPLES \
    $PROMPTS_ARG \
    --output_dir $OUTPUT_BASE/early_phase \
    2>&1 | tee $OUTPUT_BASE/log_early_phase.txt &
PID2=$!
echo "  PID: $PID2"

# -----------------------------------------------------------------------------
# 等待所有任务完成
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "All experiments launched. Waiting for completion..."
echo "  GPU 0 (Multilayer):   PID $PID0"
echo "  GPU 1 (SDXL):         PID $PID1"
echo "  GPU 2 (Early-phase):  PID $PID2"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  tail -f $OUTPUT_BASE/log_multilayer.txt"
echo "  tail -f $OUTPUT_BASE/log_sdxl.txt"
echo "  tail -f $OUTPUT_BASE/log_early_phase.txt"
echo ""

wait $PID0
echo "✓ [GPU 0] Multilayer completed"

wait $PID1
echo "✓ [GPU 1] SDXL completed"

wait $PID2
echo "✓ [GPU 2] Early-phase completed"

# -----------------------------------------------------------------------------
# 评估所有结果
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Running evaluations..."
echo "============================================================"

# 评估多层干预
echo ""
echo "[Eval] Multilayer..."
python experiments/run_multilayer.py --output_dir $OUTPUT_BASE/multilayer --eval_only \
    2>&1 | tee -a $OUTPUT_BASE/log_multilayer.txt

# 评估SDXL
# NOTE: sdxl_selfattn_sink_experiment.py does not expose an --eval_only mode;
# the evaluation pass is integrated into the generation step. Skip here.
echo ""
echo "[Eval] SDXL: built-in to generation step, skipping separate eval."

# 评估Early-phase
echo ""
echo "[Eval] Early-phase..."
python experiments/run_early_phase.py --output_dir $OUTPUT_BASE/early_phase --eval_only \
    2>&1 | tee -a $OUTPUT_BASE/log_early_phase.txt

# ImageReward评估（如果安装了）
echo ""
echo "[Eval] ImageReward on all experiments..."
if python -c "import ImageReward" 2>/dev/null; then
    python eval/eval_imagereward.py $OUTPUT_BASE/multilayer multilayer 2>&1 || true
    python eval/eval_imagereward.py $OUTPUT_BASE/sdxl_ablation sdxl 2>&1 || true
    python eval/eval_imagereward.py $OUTPUT_BASE/early_phase early 2>&1 || true
else
    echo "  ImageReward not installed, skipping..."
fi

# -----------------------------------------------------------------------------
# 生成汇总报告
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Generating Summary Report..."
echo "============================================================"

cat > $OUTPUT_BASE/SUMMARY.md << 'EOF'
# ICML Supplementary Experiments Summary

## Experiments Completed

### 1. Multilayer Intervention (GPU 0)
- Conditions: baseline, single_layer_12, multi_layer_6_12_18
- Purpose: Verify "even multi-layer removal doesn't hurt quality"

### 2. SDXL Cross-Architecture (GPU 1)
- Conditions: baseline (η=1.0), sink_removed (η=0.0)
- Purpose: Verify sink phenomenon generalizes beyond SD3

### 3. Early-Phase Intervention (GPU 2)
- Conditions: baseline, full_removal, early_only_0_20, mid_only_40_60, late_only_80_100
- Purpose: Verify "even during strongest sink phase, removal doesn't hurt"

## Key Results

EOF

# 提取关键结果
echo "### Multilayer Results" >> $OUTPUT_BASE/SUMMARY.md
if [ -f "$OUTPUT_BASE/multilayer/multilayer_stats.json" ]; then
    cat $OUTPUT_BASE/multilayer/multilayer_stats.json >> $OUTPUT_BASE/SUMMARY.md
fi
echo "" >> $OUTPUT_BASE/SUMMARY.md

echo "### SDXL Results" >> $OUTPUT_BASE/SUMMARY.md
if [ -f "$OUTPUT_BASE/sdxl_ablation/config.json" ]; then
    cat $OUTPUT_BASE/sdxl_ablation/config.json >> $OUTPUT_BASE/SUMMARY.md
fi
echo "" >> $OUTPUT_BASE/SUMMARY.md

echo "### Early-Phase Results" >> $OUTPUT_BASE/SUMMARY.md
if [ -f "$OUTPUT_BASE/early_phase/early_phase_stats.json" ]; then
    cat $OUTPUT_BASE/early_phase/early_phase_stats.json >> $OUTPUT_BASE/SUMMARY.md
fi

# -----------------------------------------------------------------------------
# 完成
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "ALL ICML EXPERIMENTS COMPLETED!"
echo "============================================================"
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Key outputs:"
echo "  📊 Multilayer:    $OUTPUT_BASE/multilayer/"
echo "  📊 SDXL:          $OUTPUT_BASE/sdxl_ablation/"
echo "  📊 Early-phase:   $OUTPUT_BASE/early_phase/"
echo "  📝 Summary:       $OUTPUT_BASE/SUMMARY.md"
echo ""
echo "Finished: $(date)"
echo "============================================================"