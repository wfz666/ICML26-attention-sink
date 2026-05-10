#!/bin/bash
#SBATCH --job-name=supple_exps_E123
#SBATCH --output=slurm-%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4



: "${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"

# ============================================================
# ICML 必做实验 - 4 GPU 并行版本
# ============================================================
#
# 时间估计（4 GPU 并行）:
#   - E1: ~10 分钟（单 GPU，只计算指标）
#   - E2: ~40 分钟（7 configs 分 4 GPU）
#   - E3: ~30 分钟（attribution + 4 modes 分 4 GPU）
#   - 总计: ~1.5 小时（vs 单 GPU 4-5 小时）
#
# GPU 分配:
#   E2:
#     GPU 0: CFG=3.0, CFG=7.5
#     GPU 1: CFG=12.0, steps=8
#     GPU 2: steps=50, euler
#     GPU 3: dpm++
#
#   E3:
#     GPU 0: Attribution analysis
#     GPU 1: mode=none, mode=text_only
#     GPU 2: mode=image_only
#     GPU 3: mode=all
#
# ============================================================

set -e

# Run from repo root
cd "$(dirname "$0")/.."

echo "============================================================"
echo "ICML 必做实验 - 4 GPU 并行版本"
echo "============================================================"
echo ""

# 检查 GPU 数量
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "检测到 $NUM_GPUS 个 GPU"
if [ "$NUM_GPUS" -lt 4 ]; then
    echo "WARNING: 少于 4 个 GPU，部分任务将排队执行"
fi
echo ""

# 检查文件
for f in src/dynamic_sink_processor.py experiments/e1_geneval_breakdown_fixed.py experiments/e2_paper_impl_multi_gpu.py experiments/e3_paper_impl_multi_gpu.py; do
    if [ ! -f "$f" ]; then
        echo "ERROR: 缺少文件 $f"
        exit 1
    fi
done
echo "✓ 所有依赖文件存在"
echo ""

# ============================================================
# E1: GenEval Tag-based Breakdown (单 GPU)
# ============================================================
echo "============================================================"
echo "[E1] GenEval Tag-based Breakdown"
echo "预计时间: ~10 分钟"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python experiments/e1_geneval_breakdown_fixed.py \
    --exp_dir results_geneval_A1_layer12_top1 \
    --output e1_breakdown_results.json \
    --device cuda

echo ""
echo "E1 完成!"
echo ""

# ============================================================
# E2 + E3 并行执行
# ============================================================
echo "============================================================"
echo "[E2 + E3] 并行执行"
echo "E2: 4 GPU 并行超参 sweep"
echo "E3: 4 GPU 并行 attribution + ablation"
echo "============================================================"

# 后台启动 E2
python experiments/e2_paper_impl_multi_gpu.py \
    --prompts_file results_geneval_A1_layer12_top1/prompts.txt \
    --output_dir results_e2_multi_gpu \
    --num_prompts 32 \
    --num_gpus 4 &
E2_PID=$!
echo "E2 启动: PID $E2_PID"

# 等待 E2 完成后再启动 E3（避免 GPU 冲突）
wait $E2_PID
echo "E2 完成!"

# 启动 E3
python experiments/e3_paper_impl_multi_gpu.py \
    --prompts_file results_geneval_A1_layer12_top1/prompts.txt \
    --output_dir results_e3_multi_gpu \
    --num_prompts_attribution 50 \
    --num_prompts_ablation 32

echo ""
echo "E3 完成!"
echo ""

# ============================================================
# 汇总
# ============================================================
echo "============================================================"
echo "所有实验完成!"
echo "============================================================"
echo ""
echo "结果文件:"
echo "  E1: e1_breakdown_results.json"
echo "  E2: results_e2_multi_gpu/e2_results.json"
echo "  E3: results_e3_multi_gpu/e3_results.json"
echo ""
echo "日志文件:"
echo "  E2: results_e2_multi_gpu/gpu*.log"
echo "  E3: results_e3_multi_gpu/gpu*.log"
