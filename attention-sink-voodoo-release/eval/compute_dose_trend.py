#!/usr/bin/env python3
"""
Dose-Response Trend Test for Sink-Specificity

Tests whether the sink-specific effect (ΔΔ) increases with masking budget k.

For each prompt i:
  d_i(k) = (HPS_sink - HPS_base) - (HPS_rand - HPS_base)

Trend test: d_i(k=50) < d_i(k=10) ? (more negative = stronger effect)

Usage:
    python compute_dose_trend.py \
        --k10_dir results_hps_k10 \
        --k50_dir results_k50_final \
        --prompts_file prompts.txt
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
from PIL import Image
from tqdm import tqdm
import torch


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42):
    """Compute bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    valid_data = data[~np.isnan(data)]
    if len(valid_data) < 2:
        return np.nan, np.nan
    
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(valid_data), size=len(valid_data))
        boot_means.append(float(valid_data[idx].mean()))
    
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


def load_images_by_index(images_dir: Path, mode: str, num_prompts: int) -> Dict[int, Image.Image]:
    """Load images as index->PIL mapping."""
    out = {}
    for i in range(num_prompts):
        p = images_dir / f"{i:03d}_{mode}.png"
        if p.exists():
            out[i] = Image.open(p).convert("RGB")
    return out


def compute_hps_scores(images: List[Image.Image], prompts: List[str]) -> np.ndarray:
    """Compute HPS-v2 scores."""
    try:
        import hpsv2
    except ImportError:
        print("ERROR: hpsv2 not installed")
        return None
    
    scores = []
    with torch.no_grad():
        for img, prompt in tqdm(zip(images, prompts), total=len(prompts), desc="HPS"):
            try:
                score = hpsv2.score(img, prompt, hps_version="v2.1")[0]
                scores.append(float(score))
            except:
                scores.append(np.nan)
    return np.array(scores)


def compute_d_i(images_dir: Path, prompts: List[str], num_prompts: int) -> np.ndarray:
    """
    Compute d_i = (HPS_sink - HPS_base) - (HPS_rand - HPS_base) for each prompt.
    """
    # Load images
    imgs_none = load_images_by_index(images_dir, "none", num_prompts)
    imgs_top = load_images_by_index(images_dir, "top_sink", num_prompts)
    imgs_rand = load_images_by_index(images_dir, "random", num_prompts)
    
    # Get common indices
    common_idx = sorted(set(imgs_none.keys()) & set(imgs_top.keys()) & set(imgs_rand.keys()) & set(range(len(prompts))))
    
    if len(common_idx) < 2:
        print(f"ERROR: Not enough common images in {images_dir}")
        return None, None
    
    print(f"Found {len(common_idx)} common indices in {images_dir}")
    
    # Compute HPS scores
    imgs_list = {
        "none": [imgs_none[i] for i in common_idx],
        "top_sink": [imgs_top[i] for i in common_idx],
        "random": [imgs_rand[i] for i in common_idx],
    }
    prompts_aligned = [prompts[i] for i in common_idx]
    
    hps_none = compute_hps_scores(imgs_list["none"], prompts_aligned)
    hps_top = compute_hps_scores(imgs_list["top_sink"], prompts_aligned)
    hps_rand = compute_hps_scores(imgs_list["random"], prompts_aligned)
    
    if hps_none is None or hps_top is None or hps_rand is None:
        return None, None
    
    # Compute d_i
    d_i = (hps_top - hps_none) - (hps_rand - hps_none)
    
    return d_i, common_idx


def main():
    parser = argparse.ArgumentParser(description="Dose-response trend test")
    parser.add_argument("--k10_dir", type=str, required=True, help="Directory for k=10 results")
    parser.add_argument("--k50_dir", type=str, required=True, help="Directory for k=50 results")
    parser.add_argument("--prompts_file", type=str, required=True, help="Prompts file")
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--output_file", type=str, default="dose_trend_results.json")
    
    args = parser.parse_args()
    
    # Load prompts
    with open(args.prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()][:args.num_prompts]
    print(f"Loaded {len(prompts)} prompts")
    
    # Compute d_i for each k
    print("\n" + "=" * 60)
    print("Computing d_i for k=10...")
    print("=" * 60)
    d_i_k10, idx_k10 = compute_d_i(Path(args.k10_dir), prompts, args.num_prompts)
    
    print("\n" + "=" * 60)
    print("Computing d_i for k=50...")
    print("=" * 60)
    d_i_k50, idx_k50 = compute_d_i(Path(args.k50_dir), prompts, args.num_prompts)
    
    if d_i_k10 is None or d_i_k50 is None:
        print("ERROR: Could not compute d_i for both k values")
        return
    
    # Find common prompts across both k values
    common_prompts = sorted(set(idx_k10) & set(idx_k50))
    print(f"\nCommon prompts across k=10 and k=50: {len(common_prompts)}")
    
    if len(common_prompts) < 10:
        print("WARNING: Very few common prompts, results may be unreliable")
    
    # Align d_i arrays to common prompts
    idx_map_k10 = {idx: i for i, idx in enumerate(idx_k10)}
    idx_map_k50 = {idx: i for i, idx in enumerate(idx_k50)}
    
    d_i_k10_aligned = np.array([d_i_k10[idx_map_k10[i]] for i in common_prompts])
    d_i_k50_aligned = np.array([d_i_k50[idx_map_k50[i]] for i in common_prompts])
    
    # Trend test: d_i(k=50) - d_i(k=10)
    # If sink-specific effect is stronger at k=50, then d_i(k=50) < d_i(k=10)
    # So we test: diff = d_i(k=50) - d_i(k=10) < 0
    diff = d_i_k50_aligned - d_i_k10_aligned
    valid = ~np.isnan(diff)
    diff_valid = diff[valid]
    
    mean_diff = float(diff_valid.mean())
    ci_low, ci_high = bootstrap_ci(diff_valid)
    
    # One-sided t-test: diff < 0?
    _, p_t_two = stats.ttest_1samp(diff_valid, popmean=0.0)
    p_t_one = p_t_two / 2.0 if mean_diff < 0 else 1.0 - (p_t_two / 2.0)
    
    # Wilcoxon
    try:
        p_w_two = wilcoxon(diff_valid).pvalue
        p_w_one = wilcoxon(diff_valid, alternative="less").pvalue
    except:
        p_w_two = np.nan
        p_w_one = np.nan
    
    ci_includes_zero = bool(ci_low <= 0 <= ci_high)
    is_trend_significant = bool((mean_diff < 0) and (not ci_includes_zero) and (p_t_one < 0.05))
    
    # Results
    results = {
        "n_common_prompts": len(common_prompts),
        "d_i_k10_mean": float(np.nanmean(d_i_k10_aligned)),
        "d_i_k50_mean": float(np.nanmean(d_i_k50_aligned)),
        "trend_diff_mean": float(mean_diff),
        "trend_diff_ci_low": float(ci_low),
        "trend_diff_ci_high": float(ci_high),
        "p_ttest_two_sided": float(p_t_two),
        "p_ttest_one_sided_less": float(p_t_one),
        "p_wilcoxon_two_sided": float(p_w_two),
        "p_wilcoxon_one_sided_less": float(p_w_one),
        "ci_includes_zero": ci_includes_zero,
        "is_trend_significant": is_trend_significant,
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("DOSE-RESPONSE TREND TEST")
    print("=" * 80)
    print(f"Hypothesis: d_i(k=50) < d_i(k=10) (stronger sink-specific effect at higher k)")
    print("-" * 80)
    print(f"N (common prompts): {len(common_prompts)}")
    print(f"Mean d_i(k=10): {np.nanmean(d_i_k10_aligned):+.4f}")
    print(f"Mean d_i(k=50): {np.nanmean(d_i_k50_aligned):+.4f}")
    print("-" * 80)
    print(f"Trend diff = d_i(k=50) - d_i(k=10)")
    print(f"  Mean: {mean_diff:+.4f}")
    print(f"  95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  p (one-sided, diff<0): t={p_t_one:.4f}, Wilcoxon={p_w_one:.4f}")
    print("-" * 80)
    
    if is_trend_significant:
        print("✓ TREND SIGNIFICANT: Sink-specific effect is significantly stronger at k=50 than k=10")
        print("  → Can claim: 'dose-dependent sink-specific degradation'")
    else:
        print("✗ Trend not significant (CI includes 0 or p >= 0.05)")
        print("  → Use descriptive: 'magnitude increases from k=10 to k=50'")
    
    print("=" * 80)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
