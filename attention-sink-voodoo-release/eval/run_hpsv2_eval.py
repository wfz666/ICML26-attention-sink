#!/usr/bin/env python
"""
HPS-v2 Evaluation Script (Production Version v3 - Final)

关键特性:
1. 严格按文件名 stem 对齐（不是简单排序截断）
2. prompts 数量必须与图像对数完全匹配（不允许静默截断）
3. 自然排序避免字典序问题
4. 支持 torch.Tensor 返回格式
5. 目录检测增强（必须有 prompts + png）
6. 错误返回 NaN，自动剔除并报告
7. 错误率 > 2% 时 abort

用法:
    pip install hpsv2
    python run_hpsv2_eval.py --a1_dir results_A1 --a2_dir results_A2
    python run_hpsv2_eval.py --auto
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy import stats
import sys
import re


# ============================================================
# 依赖检查
# ============================================================

def check_hpsv2():
    """检查 hpsv2 是否已安装"""
    try:
        import hpsv2
        return hpsv2
    except ImportError:
        print("="*60)
        print("ERROR: hpsv2 is not installed")
        print("="*60)
        print("\nPlease install it manually:")
        print("  pip install hpsv2")
        sys.exit(1)


# ============================================================
# 安全的 HPS-v2 评分（支持 torch.Tensor）
# ============================================================

def safe_hps_score(hpsv2_module, img, prompt, version="v2.1"):
    """
    安全计算 HPS-v2 分数，兼容多种返回格式
    
    支持: list, tuple, np.ndarray, torch.Tensor, float/int
    """
    out = hpsv2_module.score(img, prompt, hps_version=version)
    
    # list/tuple
    if isinstance(out, (list, tuple)):
        return float(out[0])
    
    # numpy ndarray
    if isinstance(out, np.ndarray):
        return float(out.reshape(-1)[0])
    
    # torch.Tensor
    try:
        import torch
        if isinstance(out, torch.Tensor):
            return float(out.detach().cpu().reshape(-1)[0].item())
    except ImportError:
        pass
    except Exception:
        pass
    
    # python scalar
    return float(out)


def score_images(hpsv2_module, image_paths, prompts, desc, strict=False, version="v2.1"):
    """批量计算分数，错误返回 NaN"""
    scores = []
    errors = 0
    first_error = None  # 记录第一个错误用于调试
    
    iterator = list(zip(image_paths, prompts))
    for img_path, prompt in tqdm(iterator, desc=desc):
        try:
            with Image.open(img_path) as im:
                img = im.convert("RGB").copy()
            
            score = safe_hps_score(hpsv2_module, img, prompt, version=version)
            scores.append(score)
            
        except Exception as e:
            errors += 1
            if first_error is None:
                first_error = (img_path, prompt, str(e), type(e).__name__)
            if strict:
                raise RuntimeError(f"HPSv2 failed on {img_path}: {e}") from e
            scores.append(np.nan)
    
    scores = np.array(scores, dtype=np.float64)
    
    if errors > 0:
        print(f"  [{desc}] Errors: {errors}/{len(scores)} ({errors/len(scores):.1%})")
        if first_error:
            print(f"  First error details:")
            print(f"    Image: {first_error[0]}")
            print(f"    Prompt: {first_error[1][:80]}...")
            print(f"    Error type: {first_error[3]}")
            print(f"    Error message: {first_error[2]}")
    
    return scores, errors


# ============================================================
# 统计计算
# ============================================================

def compute_paired_stats(baseline_scores, intervention_scores, n_bootstrap=1000, min_valid_ratio=0.98):
    """计算配对统计量，自动剔除 NaN"""
    mask = np.isfinite(baseline_scores) & np.isfinite(intervention_scores)
    n_original = len(baseline_scores)
    n_valid = int(np.sum(mask))
    n_dropped = n_original - n_valid
    
    if n_dropped > 0:
        print(f"  Dropped {n_dropped}/{n_original} samples with invalid scores")
    
    valid_ratio = n_valid / n_original if n_original > 0 else 0
    if valid_ratio < min_valid_ratio:
        raise RuntimeError(
            f"Too few valid samples: {n_valid}/{n_original} ({valid_ratio:.1%} < {min_valid_ratio:.0%}). "
            f"Aborting."
        )
    
    baseline = baseline_scores[mask]
    intervention = intervention_scores[mask]
    
    deltas = intervention - baseline
    delta_mean = np.mean(deltas)
    
    rng = np.random.RandomState(42)
    bootstrap_means = [
        np.mean(deltas[rng.choice(len(deltas), len(deltas), replace=True)])
        for _ in range(n_bootstrap)
    ]
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    t_stat, p_value = stats.ttest_rel(intervention, baseline)
    
    return {
        'baseline_mean': float(np.mean(baseline)),
        'baseline_std': float(np.std(baseline, ddof=1)),
        'intervention_mean': float(np.mean(intervention)),
        'intervention_std': float(np.std(intervention, ddof=1)),
        'delta_mean': float(delta_mean),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'ci_includes_zero': bool(ci_lower <= 0 <= ci_upper),
        'n_samples': n_valid,
        'n_dropped': n_dropped,
    }


# ============================================================
# 实验目录处理（增强版：检查 prompts + png）
# ============================================================

def natural_sort_key(s):
    """自然排序：1, 2, 10, 100（不是 1, 10, 100, 2）"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', str(s))]


def has_png(d: Path) -> bool:
    """检查目录是否包含 PNG 文件"""
    return any(d.glob("*.png"))


def find_experiment_dirs(base_path=Path('.')):
    """
    自动检测实验目录（增强版）
    
    必须满足:
    1. 有 baseline 目录且包含 PNG
    2. 有 intervention 目录且包含 PNG
    3. 有 prompts.txt 或 generation_prompts.txt
    """
    found = {}
    
    for d in base_path.iterdir():
        if not d.is_dir():
            continue
        
        name = d.name.lower()
        
        # 检查 baseline 目录（必须存在且有 PNG）
        has_baseline = any(
            (d / x).exists() and has_png(d / x) 
            for x in ['images_baseline', 'baseline', 'baseline_images']
        )
        
        # 检查 intervention 目录（必须存在且有 PNG）
        has_intervention = any(
            (d / x).exists() and has_png(d / x) 
            for x in ['images_intervention', 'intervention', 'intervention_images', 'images_dynamic_top1', 'images_dynamic_top5']
        )
        
        # 检查 prompts 文件
        has_prompts = (d / 'prompts.txt').exists() or (d / 'generation_prompts.txt').exists()
        
        # 必须三者都满足
        if not (has_baseline and has_intervention and has_prompts):
            continue
        
        # 识别 A1/A2/A3
        if 'a1' in name or ('layer12' in name and 'top1' in name and 'multi' not in name):
            found['a1'] = d
        elif 'a2' in name or 'multilayer' in name:
            found['a2'] = d
        elif 'a3' in name or 'top5' in name:
            found['a3'] = d
    
    return found


def load_experiment_data(exp_dir, prompts_file=None):
    """
    加载实验数据（严格对齐：按文件名 stem 匹配）
    
    不允许静默截断：prompts 数量必须与图像对数完全匹配
    """
    exp_dir = Path(exp_dir)

    # 定位目录（必须存在且包含 PNG）
    baseline_dir = None
    intervention_dir = None
    
    for bd in ['images_baseline', 'baseline', 'baseline_images']:
        cand = exp_dir / bd
        if cand.exists() and any(cand.glob("*.png")):
            baseline_dir = cand
            break
    
    for idn in ['images_intervention', 'intervention', 'intervention_images', 'images_dynamic_top1', 'images_dynamic_top5']:
        cand = exp_dir / idn
        if cand.exists() and any(cand.glob("*.png")):
            intervention_dir = cand
            break
    
    if baseline_dir is None:
        raise ValueError(f"Cannot find baseline directory with PNGs in {exp_dir}")
    if intervention_dir is None:
        raise ValueError(f"Cannot find intervention directory with PNGs in {exp_dir}")

    # 加载 prompts
    prompts = None
    candidates = []
    if prompts_file:
        candidates.append(Path(prompts_file))
        candidates.append(exp_dir / prompts_file)
    candidates += [
        exp_dir / 'prompts.txt',
        exp_dir / 'generation_prompts.txt',
        Path('generation_prompts.txt'),
    ]
    
    for pp in candidates:
        if pp and pp.exists():
            with open(pp, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            print(f"  Loaded prompts from: {pp}")
            break
    
    if prompts is None:
        raise ValueError(f"Cannot find prompts file for {exp_dir}")

    # 严格按文件名 stem 对齐
    base_files = list(baseline_dir.glob('*.png'))
    intv_files = list(intervention_dir.glob('*.png'))
    
    if len(base_files) == 0:
        raise ValueError(f"No PNG images found in {baseline_dir}")
    if len(intv_files) == 0:
        raise ValueError(f"No PNG images found in {intervention_dir}")

    base_map = {p.stem: p for p in base_files}
    intv_map = {p.stem: p for p in intv_files}
    
    common_keys = set(base_map.keys()) & set(intv_map.keys())
    
    if len(common_keys) == 0:
        raise ValueError(
            f"No matching filename stems between baseline and intervention.\n"
            f"Baseline stems (first 5): {list(base_map.keys())[:5]}\n"
            f"Intervention stems (first 5): {list(intv_map.keys())[:5]}"
        )
    
    # 警告不匹配的文件
    base_only = set(base_map.keys()) - common_keys
    intv_only = set(intv_map.keys()) - common_keys
    if base_only:
        print(f"  Warning: {len(base_only)} baseline images have no matching intervention")
    if intv_only:
        print(f"  Warning: {len(intv_only)} intervention images have no matching baseline")
    
    # 自然排序
    common_keys_sorted = sorted(common_keys, key=natural_sort_key)
    
    # 严格检查数量匹配
    if len(prompts) != len(common_keys_sorted):
        raise ValueError(
            f"Prompts count ({len(prompts)}) != paired images ({len(common_keys_sorted)}). "
            f"Refusing to auto-truncate.\n"
            f"  Baseline: {len(base_files)}, Intervention: {len(intv_files)}, Pairs: {len(common_keys_sorted)}"
        )

    baseline_images = [base_map[k] for k in common_keys_sorted]
    intervention_images = [intv_map[k] for k in common_keys_sorted]

    print(f"  Found {len(common_keys_sorted)} strictly matched image pairs")
    
    return baseline_images, intervention_images, prompts


# ============================================================
# LaTeX 生成
# ============================================================

def generate_latex_table(results):
    """生成 LaTeX 表格"""
    
    def fmt_p(p: float) -> str:
        """格式化 p-value：小于 1e-3 用科学计数法，否则保留 3 位小数"""
        return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
    
    all_include_zero = all(r['ci_includes_zero'] for r in results.values())
    
    if all_include_zero:
        caption_detail = "All 95\\% CIs include zero, indicating no significant degradation."
    else:
        caption_detail = "We report paired mean differences with 95\\% bootstrap CIs."
    
    sample_counts = [r['n_samples'] for r in results.values()]
    n_display = sample_counts[0] if len(set(sample_counts)) == 1 else "varies"
    
    latex = f"""
\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{HPS-v2 evaluation of dynamic sink interventions (GenEval, $N={n_display}$).}} 
{caption_detail}}}
\\label{{tab:hpsv2}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Condition & $N$ & $\\Delta$HPSv2 & 95\\% CI & $p$ \\\\
\\midrule"""
    
    name_map = {
        'a1': 'Single layer (L12, top-1)',
        'a2': 'Multi-layer (L6+12+18)',
        'a3': 'Stronger (L12, top-5)',
    }
    
    for exp_name in ['a1', 'a2', 'a3']:
        if exp_name in results:
            r = results[exp_name]
            latex += f"\n{name_map[exp_name]} & {r['n_samples']} & ${r['delta_mean']:+.4f}$ & $[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]$ & {fmt_p(r['p_value'])} \\\\"
    
    latex += """
\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='HPS-v2 Evaluation (Production v3 - Final)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--a1_dir', type=str, help='A1 experiment directory')
    parser.add_argument('--a2_dir', type=str, help='A2 experiment directory')
    parser.add_argument('--a3_dir', type=str, help='A3 experiment directory')
    parser.add_argument('--prompts', type=str, default='generation_prompts.txt')
    parser.add_argument('--hps_version', type=str, default='v2.1', choices=['v2.0', 'v2.1'],
                        help='HPS version to use')
    parser.add_argument('--auto', action='store_true', help='Auto-detect directories')
    parser.add_argument('--strict', action='store_true', help='Abort on any error')
    parser.add_argument('--output', type=str, default='hpsv2_results.json')
    args = parser.parse_args()
    
    print("="*60)
    print("HPS-v2 Evaluation (Production v3 - Final)")
    print("="*60)
    
    hpsv2 = check_hpsv2()
    print(f"hpsv2 loaded (version: {getattr(hpsv2, '__version__', 'unknown')})")
    print(f"Using HPS version: {args.hps_version}")
    
    # 确定实验目录
    if args.auto:
        found = find_experiment_dirs()
        if found:
            print(f"\nAuto-detected: {list(found.keys())}")
        else:
            print("\nNo valid experiments detected (need baseline+intervention+prompts)")
    else:
        found = {}
        if args.a1_dir:
            found['a1'] = Path(args.a1_dir)
        if args.a2_dir:
            found['a2'] = Path(args.a2_dir)
        if args.a3_dir:
            found['a3'] = Path(args.a3_dir)
    
    if not found:
        print("\nNo experiments to evaluate.")
        print("Usage: python run_hpsv2_eval.py --a1_dir <dir> --a2_dir <dir>")
        sys.exit(1)
    
    all_results = {}
    
    for exp_name, exp_dir in found.items():
        if not exp_dir.exists():
            print(f"\n⚠ Skipping {exp_name}: {exp_dir} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {exp_name.upper()}: {exp_dir}")
        print(f"{'='*60}")
        
        try:
            baseline_imgs, intervention_imgs, prompts = load_experiment_data(exp_dir, args.prompts)
            
            print("\nScoring baseline images...")
            baseline_scores, baseline_errors = score_images(
                hpsv2, baseline_imgs, prompts, "Baseline", strict=args.strict, version=args.hps_version
            )
            
            print("\nScoring intervention images...")
            intervention_scores, intervention_errors = score_images(
                hpsv2, intervention_imgs, prompts, "Intervention", strict=args.strict, version=args.hps_version
            )
            
            print("\nComputing statistics...")
            exp_stats = compute_paired_stats(baseline_scores, intervention_scores)
            exp_stats['baseline_errors'] = baseline_errors
            exp_stats['intervention_errors'] = intervention_errors
            
            # 保存
            output_subdir = exp_dir / 'hpsv2_eval'
            output_subdir.mkdir(exist_ok=True)
            np.save(output_subdir / 'baseline_scores.npy', baseline_scores)
            np.save(output_subdir / 'intervention_scores.npy', intervention_scores)
            with open(output_subdir / 'stats.json', 'w') as f:
                json.dump(exp_stats, f, indent=2)
            
            all_results[exp_name] = exp_stats
            
            print(f"\n{'='*40}")
            print("RESULTS")
            print(f"{'='*40}")
            print(f"Valid: {exp_stats['n_samples']} (dropped: {exp_stats['n_dropped']})")
            print(f"Baseline:      {exp_stats['baseline_mean']:.4f} ± {exp_stats['baseline_std']:.4f}")
            print(f"Intervention:  {exp_stats['intervention_mean']:.4f} ± {exp_stats['intervention_std']:.4f}")
            print(f"Δ:             {exp_stats['delta_mean']:+.4f}")
            print(f"95% CI:        [{exp_stats['ci_lower']:+.4f}, {exp_stats['ci_upper']:+.4f}]")
            print(f"p-value:       {exp_stats['p_value']:.4f}")
            print(f"{'✓' if exp_stats['ci_includes_zero'] else '⚠'} CI {'includes' if exp_stats['ci_includes_zero'] else 'excludes'} 0")
                
        except Exception as e:
            print(f"\nERROR: {e}")
            if args.strict:
                raise
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\nNo experiments evaluated successfully.")
        sys.exit(1)
    
    # 保存总结果
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    # LaTeX
    latex = generate_latex_table(all_results)
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)
    print(latex)
    
    with open('hpsv2_table.tex', 'w') as f:
        f.write(latex)
    print("\nSaved to: hpsv2_table.tex")
    
    # 汇总
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_pass = True
    for exp_name, s in all_results.items():
        status = "✓" if s['ci_includes_zero'] else "⚠"
        if not s['ci_includes_zero']:
            all_pass = False
        print(f"{status} {exp_name.upper()}: Δ={s['delta_mean']:+.4f}, CI=[{s['ci_lower']:+.4f}, {s['ci_upper']:+.4f}], p={s['p_value']:.4f}, N={s['n_samples']}")
    
    print()
    if all_pass:
        print("✓ All: CI includes 0 → sink removal does NOT hurt quality")
    else:
        print("⚠ Some CIs exclude 0 → check practical margin")


if __name__ == '__main__':
    main()
