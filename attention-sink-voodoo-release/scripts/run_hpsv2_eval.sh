#!/bin/bash
#SBATCH --job-name=hpsv2-eval
#SBATCH --output=slurm-%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1



: "${HF_TOKEN:?Set HF_TOKEN env var: export HF_TOKEN=hf_xxx}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"

# ============================================================
# HPS-v2 Evaluation for A1/A2 Experiments
# ============================================================
#
# 前提条件:
#   pip install hpsv2
#
# 使用方法:
#   sbatch run_hpsv2_eval.sh
# ============================================================

set -e

echo "============================================================"
echo "HPS-v2 Evaluation"
echo "============================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

# 检查 hpsv2（不自动安装）
python -c "import hpsv2; print('hpsv2 loaded')" || {
    echo ""
    echo "ERROR: hpsv2 not installed"
    echo "Please run: pip install hpsv2"
    exit 1
}

# 方式 1: 自动检测目录
python eval/run_hpsv2_eval.py --auto

# 方式 2: 指定目录 (取消注释并修改路径)
#python eval/run_hpsv2_eval.py \
#   --a1_dir results_geneval_A1_layer12_top1 \
#   --a2_dir results_geneval_A2_multilayer_top1 \
#   --prompts prompts/generation_prompts.txt

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
