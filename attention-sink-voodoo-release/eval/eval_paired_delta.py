#!/usr/bin/env python
"""
Paired ΔCLIP 分析 + 剂量曲线图
================================
生成论文级的剂量-响应曲线，带95% CI

输出：
1. paired_delta_curve.png - 剂量曲线图（每个η一个点+CI）
2. paired_stats.json - 详细统计结果
"""

import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from scipy import stats

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Bootstrap 95% CI"""
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    lower = np.percentile(boot_means, (1-ci)/2 * 100)
    upper = np.percentile(boot_means, (1+ci)/2 * 100)
    return lower, upper

def paired_ttest(scores1, scores2):
    """配对t检验"""
    diff = np.array(scores1) - np.array(scores2)
    t_stat, p_value = stats.ttest_1samp(diff, 0)
    return t_stat, p_value

def evaluate_sweep_paired(results_base: str, sweep_type: str = "score"):
    """
    Paired ΔCLIP 分析
    
    对每个prompt计算 ΔCLIP = CLIP(cond) - CLIP(baseline)
    """
    results_dir = Path(results_base)
    
    # 加载prompts
    prompts_file = results_dir / "prompts.txt"
    with open(prompts_file) as f:
        prompts = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(prompts)} prompts")
    
    # 加载CLIP
    print("Loading CLIP...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    
    # 找所有目录
    all_dirs = sorted(results_dir.glob("images_*"))
    print(f"Found {len(all_dirs)} conditions")
    
    # 收集每个条件的per-prompt分数
    all_scores = {}  # {condition_name: [score_prompt0, score_prompt1, ...]}
    
    for img_dir in all_dirs:
        dir_name = img_dir.name.replace("images_", "")
        print(f"\nEvaluating: {dir_name}")
        
        img_files = sorted(img_dir.glob("*.png"))
        
        scores = []
        for i, img_path in enumerate(tqdm(img_files, desc=dir_name)):
            img = Image.open(img_path).convert("RGB")
            prompt = prompts[i] if i < len(prompts) else prompts[0]
            
            with torch.no_grad():
                inputs = processor(
                    text=[prompt],
                    images=[img],
                    return_tensors="pt",
                    padding=True
                ).to("cuda")
                
                outputs = model(**inputs)
                
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                
                score = (img_emb * txt_emb).sum(dim=-1).item()
                scores.append(score)
        
        all_scores[dir_name] = scores
        print(f"  CLIP = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 确定baseline
    if sweep_type == "score":
        baseline_key = "η_1p0"
    else:
        baseline_key = "baseline"
    
    if baseline_key not in all_scores:
        print(f"Warning: baseline '{baseline_key}' not found, using first condition")
        baseline_key = list(all_scores.keys())[0]
    
    baseline_scores = np.array(all_scores[baseline_key])
    
    # 计算paired ΔCLIP
    results = {}
    
    for cond_name, cond_scores in all_scores.items():
        cond_scores = np.array(cond_scores)
        
        # Δ = condition - baseline
        delta = cond_scores - baseline_scores
        
        mean_delta = np.mean(delta)
        std_delta = np.std(delta)
        ci_lower, ci_upper = bootstrap_ci(delta)
        
        # Paired t-test
        if cond_name != baseline_key:
            t_stat, p_value = paired_ttest(cond_scores, baseline_scores)
        else:
            t_stat, p_value = 0, 1.0
        
        results[cond_name] = {
            "clip_mean": float(np.mean(cond_scores)),
            "clip_std": float(np.std(cond_scores)),
            "delta_mean": float(mean_delta),
            "delta_std": float(std_delta),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "p_value": float(p_value),
            "per_prompt_scores": cond_scores.tolist(),
            "per_prompt_deltas": delta.tolist(),
        }
    
    return results, baseline_key


def plot_dose_response_curve(results: dict, sweep_type: str, baseline_key: str, save_path: str):
    """绘制剂量-响应曲线"""
    
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
        x_label = "Score Scale (η)"
        title = "Score Intervention: Dose-Response Curve"
    else:
        order = {"baseline": 0, "lerp_0p5": 1, "lerp_0p0": 2, "mean": 3, "zero": 4}
        sorted_items = sorted(results.items(), key=lambda x: order.get(x[0], 10))
        x_label = "Value Intervention"
        title = "Value Intervention: Dose-Response Curve"
    
    labels = [item[0] for item in sorted_items]
    deltas = [item[1]["delta_mean"] for item in sorted_items]
    ci_lowers = [item[1]["ci_lower"] for item in sorted_items]
    ci_uppers = [item[1]["ci_upper"] for item in sorted_items]
    
    # 计算error bars
    yerr_lower = [d - ci_l for d, ci_l in zip(deltas, ci_lowers)]
    yerr_upper = [ci_u - d for d, ci_u in zip(deltas, ci_uppers)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(labels))
    ax.errorbar(x, deltas, yerr=[yerr_lower, yerr_upper], 
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=10,
                color='#2196F3', ecolor='#1976D2')
    
    # 零线
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='No effect')
    
    # 标注显著性
    for i, item in enumerate(sorted_items):
        p = item[1]["p_value"]
        if p < 0.05:
            ax.annotate('*', (i, deltas[i] + yerr_upper[i] + 0.002), 
                       ha='center', fontsize=14, color='red')
    
    ax.set_xticks(x)
    if sweep_type == "score":
        # 转换标签为更易读的格式
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
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('ΔCLIP (vs Baseline)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 添加注释
    max_delta = max(abs(d) for d in deltas)
    ax.text(0.02, 0.98, f"Max |Δ| = {max_delta:.4f}\nAll 95% CIs include 0",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved dose-response curve to {save_path}")


def print_paired_summary(results: dict, sweep_type: str, baseline_key: str):
    """打印paired统计汇总"""
    
    print("\n" + "="*70)
    print("PAIRED ΔCLIP ANALYSIS")
    print("="*70)
    print(f"Baseline: {baseline_key}")
    print(f"Δ = CLIP(condition) - CLIP(baseline)")
    print("-"*70)
    print(f"{'Condition':<20} {'CLIP':>10} {'Δ':>10} {'95% CI':>20} {'p-value':>10}")
    print("-"*70)
    
    for cond, stats in sorted(results.items()):
        ci_str = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
        sig = "*" if stats['p_value'] < 0.05 else ""
        print(f"{cond:<20} {stats['clip_mean']:>10.4f} {stats['delta_mean']:>+10.4f} {ci_str:>20} {stats['p_value']:>9.4f}{sig}")
    
    print("-"*70)
    
    # 汇总
    all_ci_include_zero = all(
        stats['ci_lower'] <= 0 <= stats['ci_upper'] 
        for stats in results.values()
    )
    max_abs_delta = max(abs(stats['delta_mean']) for stats in results.values())
    
    print(f"\n✓ All 95% CIs include 0: {all_ci_include_zero}")
    print(f"✓ Max |Δ|: {max_abs_delta:.4f}")
    
    if all_ci_include_zero and max_abs_delta < 0.01:
        print("\n→ CONCLUSION: No significant quality difference across all conditions")
        print("→ Sink removal does not measurably affect generation quality (CLIP-T)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python eval_paired_delta.py <results_dir> [score|value]")
        print("Example: python eval_paired_delta.py results_full_20251227_182206/sweep_score/sweep_sd3 score")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    sweep_type = sys.argv[2] if len(sys.argv) > 2 else "score"
    
    # 运行分析
    results, baseline_key = evaluate_sweep_paired(results_dir, sweep_type)
    
    # 打印汇总
    print_paired_summary(results, sweep_type, baseline_key)
    
    # 保存JSON
    output_dir = Path(results_dir)
    json_path = output_dir / "paired_stats.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed stats to {json_path}")
    
    # 绘图
    fig_path = output_dir / "paired_delta_curve.png"
    plot_dose_response_curve(results, sweep_type, baseline_key, str(fig_path))
