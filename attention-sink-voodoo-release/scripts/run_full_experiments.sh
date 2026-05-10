#!/bin/bash
#SBATCH --job-name=full-exp
#SBATCH --output=slurm-%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4



: "${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"


# =============================================================================
# Attention Sink 完整实验套件 (4 GPU并行)
# =============================================================================
# 
# 实验内容：
#   GPU 0: H1 - Sink动力学曲线（修复entropy）
#   GPU 1: H2 - Score vs Value因果对比（多个强度）
#   GPU 2: Score Sweep - 剂量曲线 (η = 1.0, 0.5, 0.1, 0.01, 0.0)
#   GPU 3: Value Sweep - 剂量曲线 (zero, mean, lerp_0.5, lerp_0.0)
#
# 预计时间：~2-3小时（取决于GPU）
# =============================================================================

set -e

# 配置
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="./results_full_${TIMESTAMP}"
NUM_SAMPLES=32
NUM_STEPS=50
SEED=42
PROMPTS_FILE="prompts/prompts_geneval_balanced_100.txt"  # 100个多样化prompts

mkdir -p $OUTPUT_BASE

# 检查prompts文件
if [ -f "$PROMPTS_FILE" ]; then
    PROMPTS_ARG="--prompts $PROMPTS_FILE"
    echo "Using prompts from: $PROMPTS_FILE"
else
    PROMPTS_ARG=""
    echo "WARNING: $PROMPTS_FILE not found, using default prompts"
fi

echo "============================================================"
echo "Attention Sink Full Experiment Suite"
echo "============================================================"
echo "Output: $OUTPUT_BASE"
echo "Samples: $NUM_SAMPLES"
echo "Steps: $NUM_STEPS"
echo "Started: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# GPU 0: H1 - Sink动力学曲线
# -----------------------------------------------------------------------------
echo ""
echo "[GPU 0] Starting H1: Sink Dynamics..."
CUDA_VISIBLE_DEVICES=0 python experiments/run_experiment.py \
    --model sd3 \
    --experiment h1 \
    --steps $NUM_STEPS \
    --num_samples $NUM_SAMPLES \
    --seed $SEED \
    $PROMPTS_ARG \
    --output_dir $OUTPUT_BASE/h1_dynamics \
    2>&1 | tee $OUTPUT_BASE/log_h1.txt &
PID0=$!

# -----------------------------------------------------------------------------
# GPU 1: H2 - Score vs Value (激进干预 η=0.01)
# -----------------------------------------------------------------------------
echo "[GPU 1] Starting H2: Score vs Value Causal Test..."
CUDA_VISIBLE_DEVICES=1 python experiments/run_experiment.py \
    --model sd3 \
    --experiment h2 \
    --steps $NUM_STEPS \
    --num_samples $NUM_SAMPLES \
    --seed $SEED \
    --score_scale 0.01 \
    --value_method zero \
    $PROMPTS_ARG \
    --output_dir $OUTPUT_BASE/h2_causal \
    2>&1 | tee $OUTPUT_BASE/log_h2.txt &
PID1=$!

# -----------------------------------------------------------------------------
# GPU 2: Score Sweep - 剂量曲线
# -----------------------------------------------------------------------------
echo "[GPU 2] Starting Score Sweep: η = [1.0, 0.5, 0.1, 0.01, 0.0]..."
CUDA_VISIBLE_DEVICES=2 python experiments/run_experiment.py \
    --model sd3 \
    --experiment sweep \
    --sweep_type score \
    --steps $NUM_STEPS \
    --num_samples $NUM_SAMPLES \
    --seed $SEED \
    $PROMPTS_ARG \
    --output_dir $OUTPUT_BASE/sweep_score \
    2>&1 | tee $OUTPUT_BASE/log_sweep_score.txt &
PID2=$!

# -----------------------------------------------------------------------------
# GPU 3: Value Sweep - 剂量曲线
# -----------------------------------------------------------------------------
echo "[GPU 3] Starting Value Sweep: methods = [zero, mean, lerp]..."
CUDA_VISIBLE_DEVICES=3 python experiments/run_experiment.py \
    --model sd3 \
    --experiment sweep \
    --sweep_type value \
    --steps $NUM_STEPS \
    --num_samples $NUM_SAMPLES \
    --seed $SEED \
    $PROMPTS_ARG \
    --output_dir $OUTPUT_BASE/sweep_value \
    2>&1 | tee $OUTPUT_BASE/log_sweep_value.txt &
PID3=$!

# -----------------------------------------------------------------------------
# 等待所有实验完成
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "All experiments started:"
echo "  GPU 0 (PID $PID0): H1 Dynamics"
echo "  GPU 1 (PID $PID1): H2 Causal"
echo "  GPU 2 (PID $PID2): Score Sweep"
echo "  GPU 3 (PID $PID3): Value Sweep"
echo "============================================================"
echo ""
echo "Waiting for completion... (this may take 2-3 hours)"
echo ""

wait $PID0
echo "✓ GPU 0 (H1) completed"

wait $PID1
echo "✓ GPU 1 (H2) completed"

wait $PID2
echo "✓ GPU 2 (Score Sweep) completed"

wait $PID3
echo "✓ GPU 3 (Value Sweep) completed"

# -----------------------------------------------------------------------------
# 质量评估
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Running Quality Evaluations..."
echo "============================================================"

# H2评估
echo ""
echo "[Eval] H2 Causal Test..."
python -m quality_metrics --results_dir $OUTPUT_BASE/h2_causal/h2_sd3 2>&1 | tee -a $OUTPUT_BASE/log_eval.txt

# Score Sweep评估
echo ""
echo "[Eval] Score Sweep..."
python -m quality_metrics --results_dir $OUTPUT_BASE/sweep_score/sweep_sd3 --sweep 2>&1 | tee -a $OUTPUT_BASE/log_eval.txt

# Value Sweep评估
echo ""
echo "[Eval] Value Sweep..."
python -m quality_metrics --results_dir $OUTPUT_BASE/sweep_value/sweep_sd3 --sweep 2>&1 | tee -a $OUTPUT_BASE/log_eval.txt

# -----------------------------------------------------------------------------
# 生成汇总报告
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Generating Summary Report..."
echo "============================================================"

cat > $OUTPUT_BASE/SUMMARY.md << 'SUMMARY_EOF'
# Attention Sink Experiment Results

## Experiment Configuration
- Model: SD3 (Stable Diffusion 3)
- Samples: 32 per condition
- Steps: 50
- Intervention Layer: 12 (middle layer)

## H1: Sink Dynamics
See: `h1_dynamics/h1_sd3/h1_curves.png`

Key metrics by layer:
- Check `h1_dynamics/h1_sd3/h1_metrics.csv`

## H2: Score vs Value Causal Test
See: `h2_causal/h2_sd3/h2_verdict.txt`

Conditions:
- none (baseline)
- score_only (η=0.01)
- value_only (zero)

## Score Sweep (Dose-Response)
See: `sweep_score/sweep_sd3/`

η values tested: 1.0, 0.5, 0.1, 0.01, 0.0

If curve is flat → strong evidence that sink removal does not hurt quality.

## Value Sweep (Dose-Response)
See: `sweep_value/sweep_sd3/`

Methods tested: zero, mean, lerp variants

## Key Findings (fill in after review)

### H1 Finding:
> [Describe phase-dependent dynamics observed]

### H2 Finding:
> [Describe causal effect or lack thereof]

### Conclusion:
> [Support/refute sink as functional component]
SUMMARY_EOF

# 添加实际结果到报告
echo "" >> $OUTPUT_BASE/SUMMARY.md
echo "## Actual Results" >> $OUTPUT_BASE/SUMMARY.md
echo "" >> $OUTPUT_BASE/SUMMARY.md

# H1 Summary
if [ -f "$OUTPUT_BASE/h1_dynamics/h1_sd3/h1_metrics.csv" ]; then
    echo "### H1 Layer Summary" >> $OUTPUT_BASE/SUMMARY.md
    echo '```' >> $OUTPUT_BASE/SUMMARY.md
    head -20 $OUTPUT_BASE/log_h1.txt | grep -A10 "H1 Summary" >> $OUTPUT_BASE/SUMMARY.md 2>/dev/null || echo "See log_h1.txt" >> $OUTPUT_BASE/SUMMARY.md
    echo '```' >> $OUTPUT_BASE/SUMMARY.md
fi

# H2 Verdict
if [ -f "$OUTPUT_BASE/h2_causal/h2_sd3/h2_verdict.txt" ]; then
    echo "" >> $OUTPUT_BASE/SUMMARY.md
    echo "### H2 Verdict" >> $OUTPUT_BASE/SUMMARY.md
    echo '```' >> $OUTPUT_BASE/SUMMARY.md
    cat $OUTPUT_BASE/h2_causal/h2_sd3/h2_verdict.txt >> $OUTPUT_BASE/SUMMARY.md
    echo '```' >> $OUTPUT_BASE/SUMMARY.md
fi

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Key outputs:"
echo "  📊 H1 Curves:     $OUTPUT_BASE/h1_dynamics/h1_sd3/h1_curves.png"
echo "  📋 H2 Verdict:    $OUTPUT_BASE/h2_causal/h2_sd3/h2_verdict.txt"
echo "  📈 Score Sweep:   $OUTPUT_BASE/sweep_score/sweep_sd3/"
echo "  📈 Value Sweep:   $OUTPUT_BASE/sweep_value/sweep_sd3/"
echo "  📝 Summary:       $OUTPUT_BASE/SUMMARY.md"
echo ""
echo "Finished: $(date)"
echo "============================================================"
























