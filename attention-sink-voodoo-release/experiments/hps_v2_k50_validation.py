#!/usr/bin/env python3
"""
HPS-v2 Validation for k=50 Counterfactual Ablation

最小版本：只跑 top_sink + random，k=50，N=64
目的：验证 CLIP-T 结果不是 metric artifact

IMPORTANT: This script REUSES the verified CounterfactualPatcher from 
ablation_counterfactual_v3.py to ensure consistent intervention mechanism.

Prerequisites:
    pip install hpsv2
    pip install image-reward  # optional

Usage:
    python hps_v2_k50_validation.py \
        --prompts_file /path/to/geneval_prompts.txt \
        --output_dir results_hps_k50 \
        --num_gpus 4
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F


# ============================================================
# HPS-v2 and ImageReward Computation
# ============================================================

def compute_hps_v2_score(images: List[Image.Image], prompts: List[str]) -> Optional[np.ndarray]:
    """
    Compute HPS-v2 (Human Preference Score v2) scores.
    Uses torch.no_grad() for determinism.
    """
    try:
        import hpsv2
    except ImportError:
        print("=" * 60)
        print("ERROR: hpsv2 not installed!")
        print("Install with: pip install hpsv2")
        print("=" * 60)
        return None
    
    scores = []
    with torch.no_grad():  # Ensure determinism
        for img, prompt in tqdm(zip(images, prompts), total=len(prompts), desc="Computing HPS-v2"):
            try:
                score = hpsv2.score(img, prompt, hps_version="v2.1")[0]
                scores.append(float(score))
            except Exception as e:
                print(f"Warning: HPS-v2 scoring failed: {e}")
                scores.append(np.nan)
    
    return np.array(scores)


def compute_image_reward_score(images: List[Image.Image], prompts: List[str], device: str = "cuda") -> Optional[np.ndarray]:
    """
    Compute ImageReward scores with model caching and eval mode.
    """
    try:
        import ImageReward as RM
    except ImportError:
        print("ImageReward not installed, skipping. Install with: pip install image-reward")
        return None
    
    # Cache model for efficiency
    if not hasattr(compute_image_reward_score, "_model"):
        model = RM.load("ImageReward-v1.0", device=device)
        model.eval()  # Set to eval mode
        compute_image_reward_score._model = model
    else:
        model = compute_image_reward_score._model
    
    scores = []
    with torch.no_grad():  # Ensure determinism
        for img, prompt in tqdm(zip(images, prompts), total=len(prompts), desc="Computing ImageReward"):
            try:
                score = model.score(prompt, img)
                scores.append(float(score))
            except Exception as e:
                print(f"Warning: ImageReward scoring failed: {e}")
                scores.append(np.nan)
    
    return np.array(scores)


def compute_clip_score(images: List[Image.Image], prompts: List[str], device: str = "cuda") -> np.ndarray:
    """Compute CLIP-T scores."""
    import clip
    
    # Cache CLIP model to avoid re-loading for each mode
    if not hasattr(compute_clip_score, "_cache"):
        compute_clip_score._cache = {}
    cache_key = (str(device), "ViT-B/32")
    if cache_key not in compute_clip_score._cache:
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        compute_clip_score._cache[cache_key] = (model, preprocess)
    else:
        model, preprocess = compute_clip_score._cache[cache_key]
    
    scores = []
    with torch.no_grad():
        for img, prompt in zip(images, prompts):
            img_input = preprocess(img).unsqueeze(0).to(device)
            text_input = clip.tokenize([prompt], truncate=True).to(device)
            
            img_features = model.encode_image(img_input)
            text_features = model.encode_text(text_input)
            
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            score = (img_features @ text_features.T).item()
            scores.append(score)
    
    return np.array(scores)


# ============================================================
# Statistics
# ============================================================

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


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni correction."""
    n = len(p_values)
    if n == 0:
        return []
    
    indexed_pvals = [(i, p) for i, p in enumerate(p_values)]
    indexed_pvals.sort(key=lambda x: x[1])
    
    adjusted = [0.0] * n
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_pvals):
        adj_p = min(1.0, p * (n - rank))
        adj_p = max(adj_p, prev_adj)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p
    
    return adjusted


def paired_diff_of_diff(baseline: np.ndarray, top: np.ndarray, rand: np.ndarray, seed: int = 42):
    """
    Compute difference-of-differences: (top - baseline) - (rand - baseline) = top - rand
    
    This tests whether top_sink effect is DIFFERENT from random effect (sink-specificity).
    One-sided test (ΔΔ < 0) is the primary test: "sink is worse than random"
    
    Returns: (mean, (ci_low, ci_high), p_ttest, p_ttest_one_sided, p_wilcoxon, p_wilcoxon_one_sided)
    """
    # d_top = top - baseline, d_rand = rand - baseline
    # d = d_top - d_rand = (top - baseline) - (rand - baseline) = top - rand
    d = (top - baseline) - (rand - baseline)
    valid = ~np.isnan(d)
    d = d[valid]
    
    if len(d) < 2:
        return np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan
    
    mean_dd = float(d.mean())
    ci_low, ci_high = bootstrap_ci(d, seed=seed)
    
    # One-sample t-test: is d different from 0? (two-sided)
    _, p_t = stats.ttest_1samp(d, popmean=0.0)
    
    # One-sided p-value: ΔΔ < 0 (sink worse than random)
    if np.isnan(p_t):
        p_t_one = np.nan
    else:
        p_t_one = p_t / 2.0 if mean_dd < 0 else 1.0 - (p_t / 2.0)
    
    # Wilcoxon signed-rank test (non-parametric)
    try:
        p_w = wilcoxon(d).pvalue
        p_w_one = wilcoxon(d, alternative="less").pvalue
    except Exception:
        p_w = np.nan
        p_w_one = np.nan
    
    return mean_dd, (float(ci_low), float(ci_high)), float(p_t), float(p_t_one), float(p_w), float(p_w_one)


# ============================================================
# Main Validation
# ============================================================

def run_hps_validation(
    prompts_file: str,
    output_dir: str,
    num_prompts: int = 64,
    target_layer: int = 12,
    top_k: int = 50,
    device: str = "cuda",
):
    """Run HPS-v2 validation for k=50 counterfactual ablation."""
    
    from diffusers import StableDiffusion3Pipeline
    
    # CRITICAL: Import the VERIFIED patcher from main experiment script
    # This avoids re-implementing attention and prevents confounded results
    try:
        from ablation_counterfactual_v3 import CounterfactualPatcher
        print("✓ Using verified CounterfactualPatcher from ablation_counterfactual_v3.py")
    except ImportError:
        try:
            from ablation_counterfactual_v2 import CounterfactualPatcher
            print("✓ Using verified CounterfactualPatcher from ablation_counterfactual_v2.py")
        except ImportError as e:
            raise RuntimeError(
                "Cannot import CounterfactualPatcher from your main script.\n"
                "Put this validation script in the same folder as ablation_counterfactual_v3.py,\n"
                "or adjust the import path."
            ) from e
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    with open(prompts_file, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    prompts = all_prompts[:num_prompts]
    print(f"Loaded {len(prompts)} prompts")
    
    # Load model
    print("Loading SD3...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to(device)
    
    # Modes: baseline, top_sink, random
    modes = ["none", "top_sink", "random"]
    all_images = {mode: [] for mode in modes}
    
    # Generate images
    # IMPORTANT: ALL modes go through the same patcher to keep code path isomorphic
    for mode in modes:
        print(f"\n[Mode: {mode}] Generating {len(prompts)} images...")
        
        # Use patcher for ALL modes (including "none") for isomorphic code path
        patcher = CounterfactualPatcher(target_layers=[target_layer], mode=mode, top_k=top_k)
        patcher.patch(pipe.transformer)
        
        for i, prompt in enumerate(tqdm(prompts, desc=f"Generating ({mode})")):
            patcher.set_random_seed_offset(i * 10000)
            gen = torch.Generator(device=device).manual_seed(1000 + i)
            img = pipe(prompt, num_inference_steps=20, generator=gen).images[0]
            all_images[mode].append(img)
            img.save(output_path / f"{i:03d}_{mode}.png")
        
        patcher.unpatch()
    
    # Compute metrics
    print("\n" + "=" * 80)
    print("COMPUTING METRICS")
    print("=" * 80)
    
    results = {
        "n_prompts": num_prompts,
        "target_layer": target_layer,
        "top_k": top_k,
        "metrics": {}
    }
    
    ablation_modes = ["top_sink", "random"]
    
    # Store raw scores for ΔΔ analysis
    raw_scores = {}
    
    # CLIP-T
    print("\n[CLIP-T]")
    baseline_clip = compute_clip_score(all_images["none"], prompts, device)
    top_clip = compute_clip_score(all_images["top_sink"], prompts, device)
    rand_clip = compute_clip_score(all_images["random"], prompts, device)
    
    raw_scores["clip_t"] = {
        "none": baseline_clip.tolist(),
        "top_sink": top_clip.tolist(),
        "random": rand_clip.tolist(),
    }
    
    results["metrics"]["clip_t"] = {}
    for mode, mode_scores in [("top_sink", top_clip), ("random", rand_clip)]:
        delta = mode_scores - baseline_clip
        ci_low, ci_high = bootstrap_ci(delta)
        _, p_value = stats.ttest_rel(baseline_clip, mode_scores)
        
        results["metrics"]["clip_t"][mode] = {
            "delta_mean": float(np.mean(delta)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_value": float(p_value),
            "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
        }
        print(f"  {mode}: Δ={np.mean(delta):+.4f}, CI=[{ci_low:+.4f}, {ci_high:+.4f}], p={p_value:.4f}")
    
    # ΔΔ test: top vs random (sink-specificity) for CLIP-T
    dd_mean, (dd_low, dd_high), p_t, p_t_one, p_w, p_w_one = paired_diff_of_diff(baseline_clip, top_clip, rand_clip)
    ci_incl_zero = bool(dd_low <= 0 <= dd_high)
    is_sink_spec = (dd_mean < 0) and (not ci_incl_zero) and (p_t_one < 0.05)
    results["metrics"]["clip_t"]["diff_of_diff_top_minus_random"] = {
        "delta_delta_mean": dd_mean,
        "ci_low": dd_low,
        "ci_high": dd_high,
        "p_ttest": p_t,
        "p_ttest_one_sided_less": p_t_one,
        "p_wilcoxon": p_w,
        "p_wilcoxon_one_sided_less": p_w_one,
        "ci_includes_zero": ci_incl_zero,
        "is_sink_specific": is_sink_spec,
    }
    print(f"  ΔΔ(top-rand): {dd_mean:+.4f}, CI=[{dd_low:+.4f}, {dd_high:+.4f}], p_1sided={p_t_one:.4f}")
    
    # HPS-v2
    print("\n[HPS-v2]")
    baseline_hps = compute_hps_v2_score(all_images["none"], prompts)
    top_hps = compute_hps_v2_score(all_images["top_sink"], prompts)
    rand_hps = compute_hps_v2_score(all_images["random"], prompts)
    
    if baseline_hps is not None and top_hps is not None and rand_hps is not None:
        raw_scores["hps_v2"] = {
            "none": baseline_hps.tolist(),
            "top_sink": top_hps.tolist(),
            "random": rand_hps.tolist(),
        }
        
        results["metrics"]["hps_v2"] = {}
        for mode, mode_scores in [("top_sink", top_hps), ("random", rand_hps)]:
            delta = mode_scores - baseline_hps
            valid = ~np.isnan(delta)
            ci_low, ci_high = bootstrap_ci(delta[valid])
            _, p_value = stats.ttest_rel(baseline_hps[valid], mode_scores[valid])
            
            results["metrics"]["hps_v2"][mode] = {
                "delta_mean": float(np.nanmean(delta)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": float(p_value),
                "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
            }
            print(f"  {mode}: Δ={np.nanmean(delta):+.4f}, CI=[{ci_low:+.4f}, {ci_high:+.4f}], p={p_value:.4f}")
        
        # ΔΔ test for HPS-v2 (CRITICAL for sink-specificity claim)
        dd_mean, (dd_low, dd_high), p_t, p_t_one, p_w, p_w_one = paired_diff_of_diff(baseline_hps, top_hps, rand_hps)
        ci_incl_zero = bool(dd_low <= 0 <= dd_high)
        is_sink_spec = (dd_mean < 0) and (not ci_incl_zero) and (p_t_one < 0.05)
        results["metrics"]["hps_v2"]["diff_of_diff_top_minus_random"] = {
            "delta_delta_mean": dd_mean,
            "ci_low": dd_low,
            "ci_high": dd_high,
            "p_ttest": p_t,
            "p_ttest_one_sided_less": p_t_one,
            "p_wilcoxon": p_w,
            "p_wilcoxon_one_sided_less": p_w_one,
            "ci_includes_zero": ci_incl_zero,
            "is_sink_specific": is_sink_spec,
        }
        print(f"  ΔΔ(top-rand): {dd_mean:+.4f}, CI=[{dd_low:+.4f}, {dd_high:+.4f}], p_1sided={p_t_one:.4f}")
    
    # ImageReward (optional)
    print("\n[ImageReward]")
    baseline_ir = compute_image_reward_score(all_images["none"], prompts, device)
    top_ir = compute_image_reward_score(all_images["top_sink"], prompts, device)
    rand_ir = compute_image_reward_score(all_images["random"], prompts, device)
    
    if baseline_ir is not None and top_ir is not None and rand_ir is not None:
        raw_scores["image_reward"] = {
            "none": baseline_ir.tolist(),
            "top_sink": top_ir.tolist(),
            "random": rand_ir.tolist(),
        }
        
        results["metrics"]["image_reward"] = {}
        for mode, mode_scores in [("top_sink", top_ir), ("random", rand_ir)]:
            delta = mode_scores - baseline_ir
            valid = ~np.isnan(delta)
            ci_low, ci_high = bootstrap_ci(delta[valid])
            _, p_value = stats.ttest_rel(baseline_ir[valid], mode_scores[valid])
            
            results["metrics"]["image_reward"][mode] = {
                "delta_mean": float(np.nanmean(delta)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": float(p_value),
                "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
            }
            print(f"  {mode}: Δ={np.nanmean(delta):+.4f}, CI=[{ci_low:+.4f}, {ci_high:+.4f}], p={p_value:.4f}")
        
        # ΔΔ test for ImageReward
        dd_mean, (dd_low, dd_high), p_t, p_t_one, p_w, p_w_one = paired_diff_of_diff(baseline_ir, top_ir, rand_ir)
        ci_incl_zero = bool(dd_low <= 0 <= dd_high)
        is_sink_spec = (dd_mean < 0) and (not ci_incl_zero) and (p_t_one < 0.05)
        results["metrics"]["image_reward"]["diff_of_diff_top_minus_random"] = {
            "delta_delta_mean": dd_mean,
            "ci_low": dd_low,
            "ci_high": dd_high,
            "p_ttest": p_t,
            "p_ttest_one_sided_less": p_t_one,
            "p_wilcoxon": p_w,
            "p_wilcoxon_one_sided_less": p_w_one,
            "ci_includes_zero": ci_incl_zero,
            "is_sink_specific": is_sink_spec,
        }
        print(f"  ΔΔ(top-rand): {dd_mean:+.4f}, CI=[{dd_low:+.4f}, {dd_high:+.4f}], p_1sided={p_t_one:.4f}")
    
    # Add raw scores to results
    results["raw_scores"] = raw_scores
    
    # Save results
    with open(output_path / "hps_validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"N = {num_prompts}, Layer = {target_layer}, k = {top_k}")
    print("-" * 100)
    print(f"{'Metric':<15} {'Mode':<20} {'Δ':>10} {'95% CI':>26} {'p':>10} {'CI∋0?':>8}")
    print("-" * 100)
    
    for metric_name, metric_results in results["metrics"].items():
        for mode, res in metric_results.items():
            if mode == "diff_of_diff_top_minus_random":
                # Handle ΔΔ separately
                continue
            ci_str = f"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]"
            ci_zero = "✓" if res.get('ci_includes_zero', res['ci_low'] <= 0 <= res['ci_high']) else "✗"
            p_val = res.get('p_value', np.nan)
            delta = res.get('delta_mean', np.nan)
            print(f"{metric_name:<15} {mode:<20} {delta:>+.4f} {ci_str:>26} {p_val:>10.4f} {ci_zero:>8}")
    
    print("=" * 100)
    
    # Print ΔΔ (sink-specificity) tests
    print("\n" + "=" * 100)
    print("SINK-SPECIFICITY TEST (ΔΔ = Δ_top - Δ_random)")
    print("=" * 100)
    print("If ΔΔ < 0 and CI excludes 0: top_sink effect is WORSE than random → sink-specific")
    print("-" * 110)
    print(f"{'Metric':<15} {'ΔΔ':>10} {'95% CI':>26} {'p(1-sided)':>12} {'Sink-Spec?':>12}")
    print("-" * 110)
    
    for metric_name, metric_results in results["metrics"].items():
        if "diff_of_diff_top_minus_random" in metric_results:
            dd = metric_results["diff_of_diff_top_minus_random"]
            ci_str = f"[{dd['ci_low']:+.4f}, {dd['ci_high']:+.4f}]"
            sink_spec = "✓ YES" if dd.get('is_sink_specific', False) else "✗ NO"
            p_one = dd.get('p_ttest_one_sided_less', dd['p_ttest'] / 2.0)
            print(f"{metric_name:<15} {dd['delta_delta_mean']:>+.4f} {ci_str:>26} {p_one:>12.4f} {sink_spec:>12}")
    
    print("=" * 110)
    
    # Interpretation
    print("\nINTERPRETATION:")
    
    # Check HPS-v2 sink-specificity
    if "hps_v2" in results["metrics"] and "diff_of_diff_top_minus_random" in results["metrics"]["hps_v2"]:
        dd_hps = results["metrics"]["hps_v2"]["diff_of_diff_top_minus_random"]
        p_one = dd_hps.get('p_ttest_one_sided_less', dd_hps['p_ttest'] / 2.0)
        if dd_hps.get('is_sink_specific', False):
            print("✓ HPS-v2: SINK-SPECIFIC effect confirmed!")
            print(f"  → ΔΔ = {dd_hps['delta_delta_mean']:+.4f}, CI excludes 0, p(one-sided) = {p_one:.4f}")
            print("  → Sink masking degrades HPS MORE than random masking of equal budget")
            print("  → Can claim: 'sink-specific quality contribution'")
        else:
            print("⚠ HPS-v2: Sink-specificity NOT confirmed")
            print(f"  → ΔΔ = {dd_hps['delta_delta_mean']:+.4f}, p(one-sided) = {p_one:.4f}")
            print("  → Cannot claim sink-specific; use: 'metric-dependent trade-off'")
    
    print(f"\nResults saved to: {output_path / 'hps_validation_results.json'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="HPS-v2 Validation for k=50 Counterfactual")
    parser.add_argument("--prompts_file", type=str, required=True,
                       help="Path to prompts file")
    parser.add_argument("--output_dir", type=str, default="results_hps_k50",
                       help="Output directory")
    parser.add_argument("--num_prompts", type=int, default=64,
                       help="Number of prompts")
    parser.add_argument("--target_layer", type=int, default=12,
                       help="Target layer")
    parser.add_argument("--top_k", type=int, default=50,
                       help="k value")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    
    args = parser.parse_args()
    
    run_hps_validation(
        prompts_file=args.prompts_file,
        output_dir=args.output_dir,
        num_prompts=args.num_prompts,
        target_layer=args.target_layer,
        top_k=args.top_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()
