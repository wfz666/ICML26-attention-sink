"""
Quality Metrics for Diffusion Generation (v2)
==============================================
修复版本：
- 使用配对检验（paired t-test）而非独立样本
- scipy可选，无则降级到bootstrap CI
- 支持sweep实验的评估

Usage:
    # H2评估
    python quality_metrics.py --results_dir ./results/h2_sd3
    
    # Sweep评估
    python quality_metrics.py --results_dir ./results/sweep_sd3 --sweep
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class QualityReport:
    """质量评估报告"""
    condition: str
    num_images: int
    clip_score_mean: float
    clip_score_std: float
    clip_scores: List[float]
    
    # 统计检验结果（与baseline对比）
    # delta = condition - baseline（负值表示质量下降）
    delta_from_baseline: Optional[float] = None
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None  # delta的95% CI下界
    ci_upper: Optional[float] = None  # delta的95% CI上界


# =============================================================================
# 1. CLIP Scorer
# =============================================================================

class CLIPScorer:
    """CLIP score计算器"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._processor = None
    
    def _load_model(self):
        if self._model is None:
            from transformers import CLIPProcessor, CLIPModel
            print(f"Loading CLIP: {self.model_name}")
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
    
    def score_batch(self, images: List[Image.Image], prompts: List[str]) -> List[float]:
        """计算一批图像-文本对的CLIP score"""
        import torch
        
        self._load_model()
        
        with torch.no_grad():
            inputs = self._processor(
                text=prompts, 
                images=images, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            outputs = self._model(**inputs)
            
            # 归一化embedding
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            scores = (image_embeds * text_embeds).sum(dim=-1)
        
        return scores.cpu().tolist()


# =============================================================================
# 2. Statistical Tests
# =============================================================================

def paired_ttest(scores1: np.ndarray, scores2: np.ndarray) -> Tuple[float, float]:
    """
    配对t检验
    
    Returns: (t_statistic, p_value)
    """
    try:
        from scipy import stats
        result = stats.ttest_rel(scores1, scores2)
        return result.statistic, result.pvalue
    except ImportError:
        # Fallback: 手动计算
        diff = scores1 - scores2
        n = len(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        se = std_diff / np.sqrt(n)
        t_stat = mean_diff / se if se > 0 else 0
        
        # 简化的p-value估计（假设正态分布）
        # 这不如scipy精确，但总比没有好
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
        return t_stat, p_value


def _normal_cdf(x: float) -> float:
    """标准正态CDF的近似"""
    import math
    return (1 + math.erf(x / math.sqrt(2))) / 2


def bootstrap_ci(
    baseline_scores: np.ndarray, 
    cond_scores: np.ndarray, 
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap置信区间
    
    计算 cond - baseline 的差值的置信区间
    （负值表示condition比baseline差）
    
    Returns: (mean_delta, ci_lower, ci_upper)
    """
    np.random.seed(42)
    n = len(baseline_scores)
    
    deltas = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        delta = np.mean(cond_scores[idx]) - np.mean(baseline_scores[idx])
        deltas.append(delta)
    
    deltas = np.array(deltas)
    mean_delta = np.mean(cond_scores) - np.mean(baseline_scores)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(deltas, alpha/2 * 100)
    ci_upper = np.percentile(deltas, (1 - alpha/2) * 100)
    
    return mean_delta, ci_lower, ci_upper


# =============================================================================
# 3. Evaluation Functions
# =============================================================================

def evaluate_condition(
    images_dir: Path,
    prompts: List[str],
    condition_name: str,
    scorer: CLIPScorer,
    batch_size: int = 8,
) -> QualityReport:
    """评估单个条件"""
    image_files = sorted(images_dir.glob("*.png"))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")
    
    # 对齐数量
    n = min(len(image_files), len(prompts))
    image_files = image_files[:n]
    prompts = prompts[:n]
    
    # 计算CLIP scores
    all_scores = []
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        batch_images = [Image.open(f).convert("RGB") for f in batch_files]
        batch_scores = scorer.score_batch(batch_images, batch_prompts)
        all_scores.extend(batch_scores)
    
    return QualityReport(
        condition=condition_name,
        num_images=len(all_scores),
        clip_score_mean=np.mean(all_scores),
        clip_score_std=np.std(all_scores),
        clip_scores=all_scores,
    )


def evaluate_h2(
    results_dir: Path,
    prompts: List[str],
    device: str = "cuda"
) -> Dict[str, QualityReport]:
    """
    评估H2实验的三个条件
    
    统计检验使用 delta = condition - baseline
    （负值表示condition比baseline差）
    """
    scorer = CLIPScorer(device=device)
    reports = {}
    
    conditions = ["none", "score_only", "value_only"]
    
    for cond in conditions:
        images_dir = results_dir / f"images_{cond}"
        if not images_dir.exists():
            print(f"Skipping {cond}: directory not found")
            continue
        
        report = evaluate_condition(images_dir, prompts, cond, scorer)
        reports[cond] = report
        print(f"{cond}: CLIP = {report.clip_score_mean:.4f} ± {report.clip_score_std:.4f}")
    
    # 与baseline对比的统计检验
    if "none" in reports:
        baseline_scores = np.array(reports["none"].clip_scores)
        
        for cond in ["score_only", "value_only"]:
            if cond not in reports:
                continue
            
            cond_scores = np.array(reports[cond].clip_scores)
            
            # 配对t检验：测试 cond - baseline 是否显著不为0
            t_stat, p_value = paired_ttest(cond_scores, baseline_scores)
            
            # Bootstrap CI for delta = cond - baseline
            mean_delta, ci_lower, ci_upper = bootstrap_ci(baseline_scores, cond_scores)
            
            reports[cond].delta_from_baseline = mean_delta
            reports[cond].p_value = p_value
            reports[cond].ci_lower = ci_lower
            reports[cond].ci_upper = ci_upper
            
            # 显示时用绝对值表示"下降幅度"
            drop = -mean_delta if mean_delta < 0 else 0
            print(f"  {cond} vs baseline: Δ={mean_delta:.4f} (drop={drop:.4f}), p={p_value:.4f}, "
                  f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return reports


def evaluate_sweep(
    results_dir: Path,
    prompts: List[str],
    sweep_type: str = "score",
    device: str = "cuda"
) -> Dict[str, QualityReport]:
    """评估sweep实验"""
    print(f"\n=== Evaluating Sweep ({sweep_type}) ===")
    print(f"Results dir: {results_dir}")
    print(f"Prompts count: {len(prompts)}")

    scorer = CLIPScorer(device=device)
    results = {}

    # 找到所有images_开头的目录
    all_dirs = sorted(results_dir.glob("images_*"))
    print(f"Found {len(all_dirs)} image directories: {[d.name for d in all_dirs]}")

    for sweep_dir in all_dirs:
        dir_name = sweep_dir.name.replace("images_", "")
        print(f"  Processing: {sweep_dir.name}")

        # 解析目录名
        if sweep_type == "score":
            # 格式: η_Xp0 或 eta_X (处理Unicode和ASCII两种情况)
            # 检查是否包含 η 或 eta 或直接是数字格式
            if dir_name.startswith("η_") or dir_name.startswith("η"):
                val_str = dir_name.replace("η_", "").replace("η", "").replace("p", ".")
            elif dir_name.startswith("eta_"):
                val_str = dir_name.replace("eta_", "").replace("p", ".")
            elif dir_name[0].isdigit() or dir_name.startswith("0"):
                # 直接是数字格式如 0p1
                val_str = dir_name.replace("p", ".")
            else:
                print(f"    Skipped: unknown score format '{dir_name}'")
                continue

            try:
                val = float(val_str)
                label = f"η={val}"
                print(f"    Parsed: val={val}, label={label}")
            except Exception as e:
                print(f"    Failed to parse '{val_str}': {e}")
                continue

        else:  # value
            # 格式: zero, mean, lerp_X, baseline
            if dir_name in ["zero", "mean", "baseline"]:
                label = dir_name
            elif dir_name.startswith("lerp_"):
                val_str = dir_name.replace("lerp_", "").replace("p", ".")
                try:
                    val = float(val_str)
                    label = f"lerp_{val}"
                except:
                    print(f"    Failed to parse lerp value")
                    continue
            else:
                print(f"    Skipped: unknown value format '{dir_name}'")
                continue
            print(f"    Label: {label}")

        report = evaluate_condition(sweep_dir, prompts, label, scorer)
        results[label] = report
        print(f"  {label}: CLIP = {report.clip_score_mean:.4f} ± {report.clip_score_std:.4f}")

    print(f"\nTotal results: {len(results)}")
    return results


# =============================================================================
# 4. Verdict & Visualization
# =============================================================================

def compute_h2_verdict(reports: Dict[str, QualityReport]) -> str:
    """
    根据质量报告判断H2是否成立

    H2: Sink的因果作用主要通过score路径，而非value路径

    delta = condition - baseline
    - 负值表示condition比baseline差（质量下降）
    - CI完全在0以下表示显著下降
    """
    if not all(k in reports for k in ["none", "score_only", "value_only"]):
        return "Incomplete data: cannot determine H2"

    baseline = reports["none"].clip_score_mean
    score_report = reports["score_only"]
    value_report = reports["value_only"]

    # 计算下降幅度（正值表示下降了多少）
    score_drop = -score_report.delta_from_baseline if score_report.delta_from_baseline else 0
    value_drop = -value_report.delta_from_baseline if value_report.delta_from_baseline else 0

    lines = [
        "="*60,
        "H2 VERDICT",
        "="*60,
        f"Baseline CLIP: {baseline:.4f}",
        "",
        "Score-only intervention:",
        f"  CLIP: {score_report.clip_score_mean:.4f}",
        f"  Δ (cond - baseline): {score_report.delta_from_baseline:.4f}",
        f"  Quality drop: {score_drop:.4f}",
        f"  p-value: {score_report.p_value:.4f}",
        f"  95% CI: [{score_report.ci_lower:.4f}, {score_report.ci_upper:.4f}]",
        "",
        "Value-only intervention:",
        f"  CLIP: {value_report.clip_score_mean:.4f}",
        f"  Δ (cond - baseline): {value_report.delta_from_baseline:.4f}",
        f"  Quality drop: {value_drop:.4f}",
        f"  p-value: {value_report.p_value:.4f}",
        f"  95% CI: [{value_report.ci_lower:.4f}, {value_report.ci_upper:.4f}]",
        "",
    ]

    # 判断显著性：CI完全在0以下表示显著下降
    score_sig = score_report.ci_upper < 0 if score_report.ci_upper is not None else False
    value_sig = value_report.ci_upper < 0 if value_report.ci_upper is not None else False

    if score_sig and not value_sig:
        lines.append("✓ SUPPORTS H2")
        lines.append("  Score intervention causes significant quality drop")
        lines.append("  Value intervention effect is not significant")
        lines.append("  → Sink primarily absorbs probability mass, not semantic info")
    elif value_sig and not score_sig:
        lines.append("✗ CONTRADICTS H2")
        lines.append("  Value intervention causes significant quality drop")
        lines.append("  Score intervention effect is not significant")
        lines.append("  → Sink may be functional infrastructure")
    elif score_sig and value_sig:
        if score_drop > value_drop * 1.5:
            lines.append("◐ PARTIALLY SUPPORTS H2")
            lines.append("  Both interventions hurt quality, but score more severely")
            lines.append(f"  Ratio: score_drop/value_drop = {score_drop/max(value_drop, 1e-6):.2f}x")
        elif value_drop > score_drop * 1.5:
            lines.append("◐ PARTIALLY CONTRADICTS H2")
            lines.append("  Both interventions hurt quality, but value more severely")
            lines.append(f"  Ratio: value_drop/score_drop = {value_drop/max(score_drop, 1e-6):.2f}x")
        else:
            lines.append("? INCONCLUSIVE")
            lines.append("  Both interventions have similar effects")
            lines.append(f"  Ratio: {score_drop/max(value_drop, 1e-6):.2f}x")
    else:
        lines.append("? INCONCLUSIVE")
        lines.append("  Neither intervention shows significant effect")
        lines.append("  Consider: stronger intervention, more samples, or different layers")

    lines.append("="*60)
    return "\n".join(lines)


def plot_h2_comparison(reports: Dict[str, QualityReport], save_path: str):
    """绘制H2对比图"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    conditions = ["none", "score_only", "value_only"]
    labels = ["Baseline", "Score Ablation", "Value Ablation"]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    available = [c for c in conditions if c in reports]
    avail_labels = [labels[conditions.index(c)] for c in available]
    avail_colors = [colors[conditions.index(c)] for c in available]

    # Bar chart with error bars
    ax = axes[0]
    means = [reports[c].clip_score_mean for c in available]
    stds = [reports[c].clip_score_std for c in available]
    bars = ax.bar(avail_labels, means, yerr=stds, capsize=5, color=avail_colors)
    ax.set_ylabel("CLIP Score")
    ax.set_title("Quality Comparison")
    ax.set_ylim(min(means) * 0.9, max(means) * 1.1)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    # Box plot
    ax = axes[1]
    data = [reports[c].clip_scores for c in available]
    bp = ax.boxplot(data, tick_labels=avail_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], avail_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("CLIP Score")
    ax.set_title("Score Distribution")

    # Effect size with CI
    ax = axes[2]
    if "none" in reports:
        effects = []
        ci_errors = []
        effect_labels = []
        effect_colors = []

        for c, label, color in zip(conditions[1:], labels[1:], colors[1:]):
            if c in reports and reports[c].delta_from_baseline is not None:
                effects.append(reports[c].delta_from_baseline)
                ci_errors.append([
                    reports[c].delta_from_baseline - reports[c].ci_lower,
                    reports[c].ci_upper - reports[c].delta_from_baseline
                ])
                effect_labels.append(label)
                effect_colors.append(color)

        if effects:
            x = range(len(effects))
            ax.bar(x, effects, color=effect_colors, alpha=0.7)
            ax.errorbar(x, effects, yerr=np.array(ci_errors).T,
                       fmt='none', color='black', capsize=5)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(effect_labels)
            ax.set_ylabel("Δ CLIP (vs Baseline)")
            ax.set_title("Effect Size with 95% CI")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def plot_sweep_curve(results: Dict[str, QualityReport], sweep_type: str, save_path: str):
    """绘制sweep曲线"""
    import matplotlib.pyplot as plt

    # 排序labels
    labels = list(results.keys())

    if sweep_type == "score":
        # 按η值排序
        def extract_eta(label):
            try:
                return float(label.replace("η=", ""))
            except:
                return 0
        labels = sorted(labels, key=extract_eta)
    else:
        # Value sweep: baseline -> lerp -> mean -> zero
        order = {"baseline": 0, "lerp_0.5": 1, "lerp_0.0": 2, "lerp_0": 2, "mean": 3, "zero": 4}
        labels = sorted(labels, key=lambda x: order.get(x, 10))

    means = [results[l].clip_score_mean for l in labels]
    stds = [results[l].clip_score_std for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(labels))
    ax.errorbar(x, means, yerr=stds, marker='o', capsize=5,
                linewidth=2, markersize=8, color='#3498db')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('CLIP Score')
    ax.set_title(f'Quality vs {sweep_type.capitalize()} Intervention Strength')
    ax.grid(True, alpha=0.3)

    # 添加baseline参考线
    if "η=1.0" in results:
        baseline = results["η=1.0"].clip_score_mean
        ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.5, label='Baseline (η=1.0)')
        ax.legend()
    elif "baseline" in results:
        baseline = results["baseline"].clip_score_mean
        ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sweep curve to {save_path}")


# =============================================================================
# 5. Main
# =============================================================================

def main():
    import argparse

    print("=== Quality Metrics Evaluation ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--sweep", action="store_true", help="Evaluate sweep experiment")
    parser.add_argument("--sweep_type", type=str, default="score", choices=["score", "value"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Args: {args}")

    results_dir = Path(args.results_dir)
    print(f"Results dir exists: {results_dir.exists()}")

    # 加载prompts
    prompts_file = results_dir / "prompts.txt"
    if prompts_file.exists():
        with open(prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Fallback
        from run_experiment import get_default_prompts
        prompts = get_default_prompts(32)

    if args.sweep:
        # Sweep评估
        results = evaluate_sweep(results_dir, prompts, args.sweep_type, args.device)

        if results:
            # 转换为可序列化格式
            report_data = {
                str(k): {"clip_mean": v.clip_score_mean, "clip_std": v.clip_score_std}
                for k, v in results.items()
            }

            with open(results_dir / "sweep_report.json", "w") as f:
                json.dump(report_data, f, indent=2)

            plot_sweep_curve(results, args.sweep_type, str(results_dir / "sweep_curve.png"))

    else:
        # H2评估
        reports = evaluate_h2(results_dir, prompts, args.device)

        if reports:
            # 保存报告
            report_data = {
                cond: {
                    "clip_mean": r.clip_score_mean,
                    "clip_std": r.clip_score_std,
                    "delta_from_baseline": r.delta_from_baseline,
                    "p_value": r.p_value,
                    "ci_lower": r.ci_lower,
                    "ci_upper": r.ci_upper,
                }
                for cond, r in reports.items()
            }

            with open(results_dir / "quality_report.json", "w") as f:
                json.dump(report_data, f, indent=2)

            # 绘图
            plot_h2_comparison(reports, str(results_dir / "h2_comparison.png"))

            # 判决
            if len(reports) >= 3:
                verdict = compute_h2_verdict(reports)
                print("\n" + verdict)

                with open(results_dir / "h2_verdict.txt", "w") as f:
                    f.write(verdict)


if __name__ == "__main__":
    main()