#!/usr/bin/env python3
"""
Compute ΔΔ (Difference-of-Differences) from existing results.

This script computes the sink-specificity test:
  ΔΔ = (top_sink - baseline) - (random - baseline) = top_sink - random

If ΔΔ < 0 and CI excludes 0: top_sink effect is WORSE than random → sink-specific

Usage:
    # From existing JSON with raw_scores
    python compute_delta_delta.py --json_file results_k50/hps_validation_results.json

    # From images directory (will recompute HPS)
    python compute_delta_delta.py \
        --images_dir results_k50 \
        --prompts_file prompts.txt
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon


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


def compute_delta_delta(
        baseline: np.ndarray,
        top: np.ndarray,
        rand: np.ndarray,
        seed: int = 42
) -> Dict:
    """
    Compute ΔΔ = (top - baseline) - (rand - baseline) = top - rand

    Returns dict with mean, CI, p-values (both two-sided and one-sided)

    One-sided test (ΔΔ < 0) is the primary test for sink-specificity:
    "top_sink degrades more than random" → ΔΔ < 0
    """
    d = (top - baseline) - (rand - baseline)  # = top - rand
    valid = ~np.isnan(d)
    d_valid = d[valid]

    if len(d_valid) < 2:
        return {
            "delta_delta_mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_ttest": np.nan,
            "p_ttest_one_sided_less": np.nan,
            "p_wilcoxon": np.nan,
            "p_wilcoxon_one_sided_less": np.nan,
            "n_valid": 0,
            "ci_includes_zero": True,
            "is_sink_specific": False,
        }

    mean_dd = float(d_valid.mean())
    ci_low, ci_high = bootstrap_ci(d_valid, seed=seed)

    # One-sample t-test: is ΔΔ different from 0? (two-sided)
    _, p_t = stats.ttest_1samp(d_valid, popmean=0.0)

    # One-sided p-value for sink-specificity hypothesis: ΔΔ < 0
    # Convert two-sided p to one-sided based on sign of mean
    if np.isnan(p_t):
        p_t_one_sided = np.nan
    else:
        if mean_dd < 0:
            p_t_one_sided = p_t / 2.0
        else:
            p_t_one_sided = 1.0 - (p_t / 2.0)

    # Wilcoxon signed-rank test (non-parametric)
    try:
        # Two-sided
        p_w = wilcoxon(d_valid).pvalue
        # One-sided (alternative='less'): ΔΔ < 0
        p_w_one_sided = wilcoxon(d_valid, alternative="less").pvalue
    except Exception:
        p_w = np.nan
        p_w_one_sided = np.nan

    ci_includes_zero = bool(ci_low <= 0 <= ci_high)
    # Sink-specific: ΔΔ < 0 AND CI excludes 0 AND one-sided p < 0.05
    is_sink_specific = bool((mean_dd < 0) and (not ci_includes_zero) and (p_t_one_sided < 0.05))

    return {
        "delta_delta_mean": float(mean_dd),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_ttest": float(p_t),
        "p_ttest_one_sided_less": float(p_t_one_sided),
        "p_wilcoxon": float(p_w),
        "p_wilcoxon_one_sided_less": float(p_w_one_sided),
        "n_valid": int(len(d_valid)),
        "ci_includes_zero": bool(ci_includes_zero),
        "is_sink_specific": bool(is_sink_specific),
    }


def from_json(json_file: str) -> Dict:
    """Compute ΔΔ from existing JSON with raw_scores."""

    with open(json_file, 'r') as f:
        data = json.load(f)

    if "raw_scores" not in data:
        print("ERROR: JSON file does not contain raw_scores")
        print("You need to re-run with the updated hps_v2_k50_validation.py")
        return None

    results = {"metrics": {}}

    for metric_name, scores in data["raw_scores"].items():
        if "none" not in scores or "top_sink" not in scores or "random" not in scores:
            print(f"Skipping {metric_name}: missing mode")
            continue

        baseline = np.array(scores["none"])
        top = np.array(scores["top_sink"])
        rand = np.array(scores["random"])

        dd = compute_delta_delta(baseline, top, rand)
        results["metrics"][metric_name] = dd

        print(f"\n[{metric_name}]")
        print(f"  ΔΔ = {dd['delta_delta_mean']:+.4f}")
        print(f"  95% CI = [{dd['ci_low']:+.4f}, {dd['ci_high']:+.4f}]")
        print(f"  p_ttest (two-sided) = {dd['p_ttest']:.4f}")
        print(f"  p_ttest (one-sided, ΔΔ<0) = {dd['p_ttest_one_sided_less']:.4f}")
        print(f"  p_wilcoxon (two-sided) = {dd['p_wilcoxon']:.4f}")
        print(f"  p_wilcoxon (one-sided, ΔΔ<0) = {dd['p_wilcoxon_one_sided_less']:.4f}")
        print(f"  CI includes 0? {dd['ci_includes_zero']}")
        print(f"  SINK-SPECIFIC? {'✓ YES' if dd['is_sink_specific'] else '✗ NO'}")

    return results


def from_images(images_dir: str, prompts_file: str, num_prompts: int = 64) -> Dict:
    """Recompute HPS and calculate ΔΔ from images."""
    from PIL import Image
    from tqdm import tqdm

    try:
        import hpsv2
    except ImportError:
        print("ERROR: hpsv2 not installed")
        return None

    import torch

    images_path = Path(images_dir)

    # Load prompts
    with open(prompts_file, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    prompts = all_prompts[:num_prompts]

    # Load images by index
    def load_by_index(mode):
        imgs = {}
        for i in range(num_prompts):
            p = images_path / f"{i:03d}_{mode}.png"
            if p.exists():
                imgs[i] = Image.open(p).convert("RGB")
        return imgs

    imgs_none = load_by_index("none")
    imgs_top = load_by_index("top_sink")
    imgs_rand = load_by_index("random")

    # Get common indices
    common_idx = sorted(set(imgs_none.keys()) & set(imgs_top.keys()) & set(imgs_rand.keys()))
    print(f"Found {len(common_idx)} common indices")

    if len(common_idx) < 2:
        print("ERROR: Not enough common images")
        return None

    # Compute HPS
    def compute_hps(imgs_dict, idxs):
        scores = []
        with torch.no_grad():
            for i in tqdm(idxs, desc="HPS"):
                try:
                    score = hpsv2.score(imgs_dict[i], prompts[i], hps_version="v2.1")[0]
                    scores.append(float(score))
                except:
                    scores.append(np.nan)
        return np.array(scores)

    print("\nComputing HPS scores...")
    hps_none = compute_hps(imgs_none, common_idx)
    hps_top = compute_hps(imgs_top, common_idx)
    hps_rand = compute_hps(imgs_rand, common_idx)

    # Compute ΔΔ
    dd = compute_delta_delta(hps_none, hps_top, hps_rand)

    print(f"\n[HPS-v2 ΔΔ Result]")
    print(f"  ΔΔ = {dd['delta_delta_mean']:+.4f}")
    print(f"  95% CI = [{dd['ci_low']:+.4f}, {dd['ci_high']:+.4f}]")
    print(f"  p_ttest (two-sided) = {dd['p_ttest']:.4f}")
    print(f"  p_ttest (one-sided, ΔΔ<0) = {dd['p_ttest_one_sided_less']:.4f}")
    print(f"  p_wilcoxon (two-sided) = {dd['p_wilcoxon']:.4f}")
    print(f"  p_wilcoxon (one-sided, ΔΔ<0) = {dd['p_wilcoxon_one_sided_less']:.4f}")
    print(f"  SINK-SPECIFIC? {'✓ YES' if dd['is_sink_specific'] else '✗ NO'}")

    return {"metrics": {"hps_v2": dd}}


def main():
    parser = argparse.ArgumentParser(description="Compute ΔΔ (sink-specificity test)")
    parser.add_argument("--json_file", type=str, default=None,
                        help="JSON file with raw_scores")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Directory with images (alternative to json_file)")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Prompts file (required if using images_dir)")
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--output_file", type=str, default="delta_delta_results.json",
                        help="Output JSON file")

    args = parser.parse_args()

    if args.json_file:
        results = from_json(args.json_file)
    elif args.images_dir and args.prompts_file:
        results = from_images(args.images_dir, args.prompts_file, args.num_prompts)
    else:
        print("ERROR: Provide either --json_file or (--images_dir and --prompts_file)")
        return

    if results:
        # Print summary
        print("\n" + "=" * 100)
        print("SINK-SPECIFICITY SUMMARY")
        print("=" * 100)
        print("ΔΔ = Δ_top - Δ_random = (top-baseline) - (random-baseline)")
        print("Hypothesis: sink masking is WORSE than random → ΔΔ < 0 (one-sided test)")
        print("Sink-specific if: ΔΔ < 0 AND CI excludes 0 AND p_one_sided < 0.05")
        print("-" * 100)
        print(f"{'Metric':<15} {'ΔΔ':>8} {'95% CI':>22} {'p(1-sided)':>12} {'Sink-Spec?':>12}")
        print("-" * 100)

        for metric, dd in results["metrics"].items():
            sink_spec = "✓ YES" if dd['is_sink_specific'] else "✗ NO"
            ci_str = f"[{dd['ci_low']:+.4f}, {dd['ci_high']:+.4f}]"
            print(
                f"{metric:<15} {dd['delta_delta_mean']:>+.4f} {ci_str:>22} {dd['p_ttest_one_sided_less']:>12.4f} {sink_spec:>12}")

        print("=" * 100)

        # Save
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()