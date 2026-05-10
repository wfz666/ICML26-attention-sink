#!/usr/bin/env python3
"""
ImageReward Evaluation Script (Standalone)

This script ONLY computes ImageReward scores on existing images.
Use this when ImageReward requires a different environment.

Usage:
    # Activate ImageReward environment first
    conda activate imagereward_env
    
    python compute_imagereward.py \
        --images_dir results_hps_k50 \
        --prompts_file /path/to/prompts.txt \
        --num_prompts 64 \
        --output_file imagereward_results.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from scipy import stats
from PIL import Image
from tqdm import tqdm

import torch


def compute_image_reward_score(
    images: List[Image.Image], 
    prompts: List[str], 
    device: str = "cuda"
) -> Optional[np.ndarray]:
    """Compute ImageReward scores with model caching and eval mode."""
    try:
        import ImageReward as RM
    except ImportError:
        print("=" * 60)
        print("ERROR: ImageReward not installed!")
        print("Install with: pip install image-reward")
        print("=" * 60)
        return None
    
    # Cache model
    if not hasattr(compute_image_reward_score, "_model"):
        print("Loading ImageReward model...")
        model = RM.load("ImageReward-v1.0", device=device)
        model.eval()
        compute_image_reward_score._model = model
    else:
        model = compute_image_reward_score._model
    
    scores = []
    with torch.no_grad():
        for img, prompt in tqdm(zip(images, prompts), total=len(prompts), desc="ImageReward"):
            try:
                score = model.score(prompt, img)
                scores.append(float(score))
            except Exception as e:
                print(f"Warning: scoring failed: {e}")
                scores.append(np.nan)
    
    return np.array(scores)


def load_images_by_index(images_dir: Path, mode: str, num_prompts: int) -> Dict[int, Image.Image]:
    """Load images as index->PIL mapping to avoid misalignment."""
    out = {}
    for i in range(num_prompts):
        p = images_dir / f"{i:03d}_{mode}.png"
        if p.exists():
            out[i] = Image.open(p).convert("RGB")
    return out


def get_aligned_pairs(
    prompts: List[str],
    baseline_imgs: Dict[int, Image.Image],
    mode_imgs: Dict[int, Image.Image],
):
    """Get aligned (baseline, mode, prompt) pairs based on common indices."""
    idxs = sorted(set(baseline_imgs.keys()) & set(mode_imgs.keys()) & set(range(len(prompts))))
    base = [baseline_imgs[i] for i in idxs]
    mod = [mode_imgs[i] for i in idxs]
    pr = [prompts[i] for i in idxs]
    return base, mod, pr, idxs


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


def main():
    parser = argparse.ArgumentParser(description="Compute ImageReward on existing images")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="Directory containing generated images")
    parser.add_argument("--prompts_file", type=str, required=True,
                       help="Path to prompts file")
    parser.add_argument("--num_prompts", type=int, default=64,
                       help="Number of prompts to evaluate")
    parser.add_argument("--output_file", type=str, default="imagereward_results.json",
                       help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    parser.add_argument("--modes", type=str, default="top_sink,random,bottom_sink",
                       help="Comma-separated modes to evaluate")
    
    args = parser.parse_args()
    
    images_path = Path(args.images_dir)
    
    # Load prompts
    with open(args.prompts_file, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    prompts = all_prompts[:args.num_prompts]
    print(f"Loaded {len(prompts)} prompts")
    
    # Load baseline images
    imgs_baseline = load_images_by_index(images_path, "none", args.num_prompts)
    print(f"Loaded {len(imgs_baseline)} baseline images")
    
    if len(imgs_baseline) == 0:
        print("ERROR: No baseline images found!")
        return
    
    # Parse modes
    modes = [m.strip() for m in args.modes.split(",")]
    
    # Results
    results = {
        "images_dir": str(images_path),
        "num_prompts": args.num_prompts,
        "modes": {}
    }
    
    print("\n" + "=" * 70)
    print("COMPUTING IMAGEREWARD")
    print("=" * 70)
    
    for mode in modes:
        print(f"\n[Mode: {mode}]")
        
        # Load mode images
        imgs_mode = load_images_by_index(images_path, mode, args.num_prompts)
        print(f"  Loaded {len(imgs_mode)} images")
        
        if len(imgs_mode) == 0:
            print(f"  Skipping: no images found")
            continue
        
        # Get aligned pairs
        base_imgs, mode_imgs, aligned_prompts, idxs = get_aligned_pairs(
            prompts, imgs_baseline, imgs_mode
        )
        print(f"  Aligned pairs: {len(idxs)}")
        
        if len(idxs) < 2:
            print(f"  Skipping: not enough pairs")
            continue
        
        # Compute scores
        baseline_scores = compute_image_reward_score(base_imgs, aligned_prompts, args.device)
        mode_scores = compute_image_reward_score(mode_imgs, aligned_prompts, args.device)
        
        if baseline_scores is None or mode_scores is None:
            print(f"  Skipping: scoring failed")
            continue
        
        # Compute delta
        delta = mode_scores - baseline_scores
        valid = ~np.isnan(delta)
        
        if valid.sum() < 2:
            print(f"  Skipping: not enough valid scores")
            continue
        
        ci_low, ci_high = bootstrap_ci(delta[valid])
        _, p_value = stats.ttest_rel(baseline_scores[valid], mode_scores[valid])
        
        results["modes"][mode] = {
            "n_pairs": int(valid.sum()),
            "delta_mean": float(np.nanmean(delta)),
            "delta_std": float(np.nanstd(delta)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_value": float(p_value),
            "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
            "baseline_mean": float(np.nanmean(baseline_scores)),
            "mode_mean": float(np.nanmean(mode_scores)),
        }
        
        ci_zero = "✓" if ci_low <= 0 <= ci_high else "✗"
        print(f"  Δ = {np.nanmean(delta):+.4f}")
        print(f"  95% CI = [{ci_low:+.4f}, {ci_high:+.4f}]")
        print(f"  p = {p_value:.4f}")
        print(f"  CI∋0? {ci_zero}")
    
    # Save results
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<15} {'Δ':>10} {'95% CI':>24} {'p':>10} {'CI∋0?':>8}")
    print("-" * 70)
    
    for mode, res in results["modes"].items():
        ci_str = f"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]"
        ci_zero = "✓" if res['ci_includes_zero'] else "✗"
        print(f"{mode:<15} {res['delta_mean']:>+.4f} {ci_str:>24} {res['p_value']:>10.4f} {ci_zero:>8}")
    
    print("=" * 70)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
