#!/bin/bash
#SBATCH --job-name=supple_exps_round2
#SBATCH --output=slurm-%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4



: "${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"

# ============================================================
# 补充实验：SDXL + FID Calibration
# ============================================================
#
# 实验 1: SDXL Cross-Attention Sink Intervention
#   - 验证 text-sink story 在 SDXL 上的表现
#   - 对比 SD3 joint attention 的结果
#   - 包含 2 个 sanity check：no-op 和 intervention verification
#
# 实验 2: FID Calibration Baseline
#   - 给 FID shift 提供参照系
#   - 对比 seed/CFG/steps/scheduler 变化的影响
#
# ============================================================

set -e

# Run from repo root
cd "$(dirname "$0")/.."

# This script reuses the prompt set from a prior dynamic-sink run; you must
# either run scripts/run_geneval_experiments.sh first (which writes
# results_geneval_A1_layer12_top1/prompts.txt) or override PROMPTS_FILE below.
PROMPTS_FILE="${PROMPTS_FILE:-results_geneval_A1_layer12_top1/prompts.txt}"

echo "============================================================"
echo "补充实验：SDXL + FID Calibration"
echo "============================================================"
echo ""

# 检查文件
for f in experiments/sdxl_sink_experiment.py experiments/fid_calibration_experiment.py; do
    if [ ! -f "$f" ]; then
        echo "ERROR: 缺少文件 $f"
        exit 1
    fi
done
echo "✓ 依赖文件存在"
echo ""

# ============================================================
# 实验 1: SDXL Cross-Attention Sink Intervention
# ============================================================
echo "============================================================"
echo "[实验 1] SDXL Cross-Attention Sink Intervention"
echo "目标：验证 text-sink story 在 explicit cross-attention 上的表现"
echo "预计时间: ~30 分钟"
echo ""
echo "Sanity checks:"
echo "  1. No-op = pixel-identical"
echo "  2. Intervention reduces sink mass by >100x"
echo "============================================================"

python experiments/sdxl_sink_experiment.py \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir results_sdxl_sink \
    --num_prompts 32 \
    --device cuda

echo ""
echo "SDXL 实验完成!"
echo ""

# ============================================================
# 实验 2: FID Calibration Baseline
# ============================================================
echo "============================================================"
echo "[实验 2] FID Calibration Baseline"
echo "目标：给 FID shift 提供参照系"
echo "预计时间: ~1.5 小时 (生成 7 组 × 100 images)"
echo ""
echo "对比项:"
echo "  - Seed variation"
echo "  - CFG ±1"
echo "  - Steps -5/-10"
echo "  - Scheduler: default → Euler"
echo "============================================================"

python experiments/fid_calibration_experiment.py \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir results_fid_calibration \
    --num_prompts 100 \
    --model sd3 \
    --device cuda

echo ""
echo "FID Calibration 完成!"
echo ""

# ============================================================
# 汇总
# ============================================================
echo "============================================================"
echo "所有补充实验完成!"
echo "============================================================"
echo ""
echo "结果文件:"
echo "  SDXL: results_sdxl_sink/sdxl_sink_results.json"
echo "  FID:  results_fid_calibration/fid_calibration_results.json"
echo ""
echo "预期结论:"
echo "  SDXL: CI 包含零 → text-sink intervention 在 cross-attn 上也无害"
echo "  FID:  Intervention FID 与 seed/CFG 变化可比 → 不是异常值"
