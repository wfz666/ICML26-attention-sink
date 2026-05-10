#!/usr/bin/env python
"""
ImageReward 评估脚本
====================
在现有sweep结果上计算ImageReward分数

安装: pip install image-reward

Usage:
    python eval_imagereward.py results_full_20251227_182206/sweep_score/sweep_sd3 score
    python eval_imagereward.py results_full_20251227_182206/sweep_value/sweep_sd3 value
"""

import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_imagereward():
    """加载ImageReward模型"""
    try:
        import ImageReward as RM
        print("Loading ImageReward model...")
        model = RM.load("ImageReward-v1.0")
        return model
    except ImportError:
        print("请先安装: pip install image-reward")
        raise


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Bootstrap 95% CI"""
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1-ci)/2 * 100)
    upper = np.percentile(boot_means, (1+ci)/2 * 100)
    return lower, upper


def evaluate_imagereward(results_dir: str, sweep_type: str = "score"):
    """
    在sweep结果上计算ImageReward
    """
    results_path = Path(results_dir)
    
    # 加载prompts
    prompts_file = results_path / "prompts.txt"
    with open(prompts_file) as f:
        prompts = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(prompts)} prompts")
    
    # 加载模型
    model = load_imagereward()
    
    # 找所有目录
    all_dirs = sorted(results_path.glob("images_*"))
    print(f"Found {len(all_dirs)} conditions")
    
    # 收集每个条件的分数
    all_scores = {}
    
    for img_dir in all_dirs:
        dir_name = img_dir.name.replace("images_", "")
        print(f"\nEvaluating: {dir_name}")
        
        img_files = sorted(img_dir.glob("*.png"))
        
        # 强制 prompt 与图片一一对应，否则报错
        if len(img_files) != len(prompts):
            raise RuntimeError(
                f"Prompt/image count mismatch in {img_dir}: "
                f"{len(img_files)} images vs {len(prompts)} prompts. "
                "Ensure per-prompt generation saved exactly one image per prompt."
            )
        
        scores = []
        for i, img_path in enumerate(tqdm(img_files, desc=dir_name)):
            prompt = prompts[i]
            
            # ImageReward评分
            score = model.score(prompt, str(img_path))
            scores.append(score)
        
        all_scores[dir_name] = scores
        print(f"  ImageReward = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 确定baseline - 支持多种目录命名
    # Prefer common baseline names
    preferred = ["baseline", "η_1p0", "eta_1p0", "none"]
    baseline_key = None
    for k in preferred:
        if k in all_scores:
            baseline_key = k
            break
    if baseline_key is None:
        baseline_key = list(all_scores.keys())[0]
        print(f"Warning: No standard baseline found, using '{baseline_key}'")
    
    baseline_scores = np.array(all_scores[baseline_key])
    
    # 计算paired delta
    results = {}
    
    print("\n" + "="*70)
    print("PAIRED ΔIMAGEREWARD ANALYSIS")
    print("="*70)
    print(f"Baseline: {baseline_key}")
    print("-"*70)
    print(f"{'Condition':<20} {'Score':>10} {'Δ':>10} {'95% CI':>22} {'p-value':>10}")
    print("-"*70)
    
    for cond_name, cond_scores in all_scores.items():
        cond_scores = np.array(cond_scores)
        delta = cond_scores - baseline_scores
        
        mean_score = np.mean(cond_scores)
        mean_delta = np.mean(delta)
        ci_lower, ci_upper = bootstrap_ci(delta)
        
        # Paired t-test (with scipy fallback)
        try:
            from scipy import stats
            if cond_name != baseline_key:
                _, p_value = stats.ttest_1samp(delta, 0)
            else:
                p_value = 1.0
        except ImportError:
            # Fallback: 使用bootstrap估计p-value
            if cond_name != baseline_key:
                # 简单的permutation test
                n_perm = 1000
                count = 0
                for _ in range(n_perm):
                    perm_delta = delta * np.random.choice([-1, 1], size=len(delta))
                    if abs(np.mean(perm_delta)) >= abs(mean_delta):
                        count += 1
                p_value = count / n_perm
            else:
                p_value = 1.0
        
        results[cond_name] = {
            "score_mean": float(mean_score),
            "score_std": float(np.std(cond_scores)),
            "delta_mean": float(mean_delta),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "p_value": float(p_value),
            "per_prompt_scores": cond_scores.tolist(),
        }
        
        ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        print(f"{cond_name:<20} {mean_score:>10.4f} {mean_delta:>+10.4f} {ci_str:>22} {p_value:>10.4f}")
    
    print("-"*70)
    
    # 检查是否所有CI包含0
    all_ci_include_zero = all(
        r['ci_lower'] <= 0 <= r['ci_upper'] 
        for r in results.values()
    )
    max_abs_delta = max(abs(r['delta_mean']) for r in results.values())
    
    print(f"\n✓ All 95% CIs include 0: {all_ci_include_zero}")
    print(f"✓ Max |Δ|: {max_abs_delta:.4f}")
    
    if all_ci_include_zero:
        print("\n→ CONCLUSION: No significant quality difference (ImageReward)")
    
    # 保存结果
    json_path = results_path / "imagereward_stats.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {json_path}")
    
    # 绘图
    plot_imagereward_curve(results, sweep_type, baseline_key, 
                           str(results_path / "imagereward_curve.png"))
    
    return results


def plot_imagereward_curve(results: dict, sweep_type: str, baseline_key: str, save_path: str):
    """绘制ImageReward剂量曲线"""
    
    # 排序
    if sweep_type == "score":
        def sort_key(x):
            name = x[0]
            val_str = name.replace("η_", "").replace("p", ".")
            try:
                return float(val_str)
            except:
                return 999
        sorted_items = sorted(results.items(), key=sort_key)
    else:
        order = {"baseline": 0, "lerp_0p5": 1, "lerp_0p0": 2, "mean": 3, "zero": 4}
        sorted_items = sorted(results.items(), key=lambda x: order.get(x[0], 10))
    
    labels = [item[0] for item in sorted_items]
    deltas = [item[1]["delta_mean"] for item in sorted_items]
    ci_lowers = [item[1]["ci_lower"] for item in sorted_items]
    ci_uppers = [item[1]["ci_upper"] for item in sorted_items]
    
    yerr_lower = [d - ci_l for d, ci_l in zip(deltas, ci_lowers)]
    yerr_upper = [ci_u - d for d, ci_u in zip(deltas, ci_uppers)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(labels))
    ax.errorbar(x, deltas, yerr=[yerr_lower, yerr_upper], 
                fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=10,
                color='#4CAF50', ecolor='#388E3C', label='ImageReward')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xticks(x)
    if sweep_type == "score":
        readable_labels = []
        for l in labels:
            val_str = l.replace("η_", "").replace("p", ".")
            try:
                val = float(val_str)
                readable_labels.append(f"η={val}")
            except:
                readable_labels.append(l)
        ax.set_xticklabels(readable_labels, rotation=45, ha='right')
    else:
        ax.set_xticklabels(labels, rotation=45, ha='right')
    
    ax.set_xlabel('Intervention Strength', fontsize=12)
    ax.set_ylabel('ΔImageReward (vs Baseline)', fontsize=12)
    ax.set_title('ImageReward: Dose-Response Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def create_multi_metric_comparison(results_dir: str, sweep_type: str = "score"):
    """
    创建多指标对比表（CLIP + ImageReward）
    """
    results_path = Path(results_dir)
    
    # 加载两个指标的结果
    clip_file = results_path / "paired_stats.json"
    ir_file = results_path / "imagereward_stats.json"
    
    if not clip_file.exists() or not ir_file.exists():
        print("请先运行 eval_paired_delta.py 和 eval_imagereward.py")
        return
    
    with open(clip_file) as f:
        clip_results = json.load(f)
    with open(ir_file) as f:
        ir_results = json.load(f)
    
    print("\n" + "="*80)
    print("MULTI-METRIC COMPARISON (Table 1)")
    print("="*80)
    print(f"{'Condition':<15} {'CLIP':>10} {'ΔCLIP':>10} {'IR':>10} {'ΔIR':>10} {'Both CI∋0':>12}")
    print("-"*80)
    
    for cond in clip_results.keys():
        if cond not in ir_results:
            continue
        
        clip_mean = clip_results[cond].get("clip_mean", clip_results[cond].get("score_mean", 0))
        clip_delta = clip_results[cond]["delta_mean"]
        clip_ci = (clip_results[cond]["ci_lower"], clip_results[cond]["ci_upper"])
        
        ir_mean = ir_results[cond]["score_mean"]
        ir_delta = ir_results[cond]["delta_mean"]
        ir_ci = (ir_results[cond]["ci_lower"], ir_results[cond]["ci_upper"])
        
        clip_includes_zero = clip_ci[0] <= 0 <= clip_ci[1]
        ir_includes_zero = ir_ci[0] <= 0 <= ir_ci[1]
        both_ok = "✓" if (clip_includes_zero and ir_includes_zero) else "✗"
        
        print(f"{cond:<15} {clip_mean:>10.4f} {clip_delta:>+10.4f} {ir_mean:>10.4f} {ir_delta:>+10.4f} {both_ok:>12}")
    
    print("-"*80)
    print("\n✓ = 95% CI includes 0 for both metrics")
    
    # 保存为Table 1格式
    table_path = results_path / "table1_multi_metric.txt"
    with open(table_path, "w") as f:
        f.write("Multi-Metric Comparison (Table 1)\n")
        f.write("="*60 + "\n")
        f.write(f"{'Condition':<15} {'ΔCLIP':>12} {'ΔImageReward':>15}\n")
        f.write("-"*60 + "\n")
        for cond in clip_results.keys():
            if cond not in ir_results:
                continue
            clip_delta = clip_results[cond]["delta_mean"]
            ir_delta = ir_results[cond]["delta_mean"]
            f.write(f"{cond:<15} {clip_delta:>+12.4f} {ir_delta:>+15.4f}\n")
    
    print(f"\nSaved Table 1 to {table_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python eval_imagereward.py <results_dir> [score|value]")
        print("  python eval_imagereward.py <results_dir> [score|value] --table")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    sweep_type = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ["score", "value"] else "score"
    
    if "--table" in sys.argv:
        create_multi_metric_comparison(results_dir, sweep_type)
    else:
        evaluate_imagereward(results_dir, sweep_type)
        print("\n" + "="*60)
        print("Next: Create multi-metric table")
        print(f"  python eval_imagereward.py {results_dir} {sweep_type} --table")
        print("="*60)
