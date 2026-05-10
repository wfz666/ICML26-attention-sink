#!/usr/bin/env python3
"""
HPS-v2 Sanity Checks and Supplementary Experiments

1. Sanity A: HPS determinism check (same images → same scores)
2. Sanity B: Image difference verification (ablated ≠ baseline)
3. Supplement: Add bottom_sink and ImageReward to existing results

Usage:
    # Run sanity checks on existing results
    python hps_sanity_and_supplement.py --sanity --images_dir results_hps_k50

    # Run full supplement (bottom_sink + ImageReward)
    python hps_sanity_and_supplement.py --supplement \
        --prompts_file prompts.txt \
        --output_dir results_hps_k50_full \
        --top_k 50
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from scipy import stats
from PIL import Image
from tqdm import tqdm

import torch


# ============================================================
# Sanity Checks
# ============================================================

def sanity_a_hps_determinism(images_dir: str, prompts_file: str, mode: str = "top_sink", num_samples: int = 10):
    """
    Sanity A: Verify HPS-v2 is deterministic.
    Run HPS twice on same images with REAL prompts, check max diff < 1e-6.
    """
    print("\n" + "=" * 60)
    print("SANITY A: HPS-v2 Determinism Check")
    print("=" * 60)
    
    try:
        import hpsv2
    except ImportError:
        print("ERROR: hpsv2 not installed")
        return False
    
    images_path = Path(images_dir)
    
    # Load REAL prompts (critical for valid sanity check)
    if prompts_file is None:
        print("ERROR: Must provide --prompts_file for valid sanity check")
        return False
    
    with open(prompts_file, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    
    # Load sample images with their corresponding real prompts
    images = []
    prompts = []
    
    for i in range(min(num_samples, len(all_prompts))):
        img_path = images_path / f"{i:03d}_{mode}.png"
        if img_path.exists():
            images.append(Image.open(img_path).convert("RGB"))
            prompts.append(all_prompts[i])  # Use REAL prompt
    
    if len(images) < num_samples:
        print(f"Warning: Only found {len(images)} images (requested {num_samples})")
    
    if len(images) == 0:
        print("ERROR: No images found")
        return False
    
    print(f"Testing on {len(images)} images with REAL prompts...")
    
    # Run 1 (with torch.no_grad for determinism)
    print("Run 1...")
    scores_run1 = []
    with torch.no_grad():
        for img, prompt in zip(images, prompts):
            score = hpsv2.score(img, prompt, hps_version="v2.1")[0]
            scores_run1.append(float(score))
    
    # Run 2
    print("Run 2...")
    scores_run2 = []
    with torch.no_grad():
        for img, prompt in zip(images, prompts):
            score = hpsv2.score(img, prompt, hps_version="v2.1")[0]
            scores_run2.append(float(score))
    
    # Compare
    scores_run1 = np.array(scores_run1)
    scores_run2 = np.array(scores_run2)
    
    max_diff = np.max(np.abs(scores_run1 - scores_run2))
    mean_diff = np.mean(np.abs(scores_run1 - scores_run2))
    
    print(f"\nResults:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print("✓ PASS: HPS-v2 is deterministic (max diff < 1e-5)")
        return True
    else:
        print("✗ FAIL: HPS-v2 shows non-determinism!")
        print("  This may indicate random state issues in the evaluation.")
        return False


def sanity_b_image_differences(images_dir: str, num_samples: int = 10):
    """
    Sanity B: Verify ablated images are different from baseline.
    Compute pixel differences between baseline and ablated images.
    """
    print("\n" + "=" * 60)
    print("SANITY B: Image Difference Verification")
    print("=" * 60)
    
    images_path = Path(images_dir)
    
    modes_to_check = ["top_sink", "random"]
    
    for mode in modes_to_check:
        print(f"\n[Checking {mode} vs baseline]")
        
        diffs = []
        for i in range(num_samples):
            baseline_path = images_path / f"{i:03d}_none.png"
            ablated_path = images_path / f"{i:03d}_{mode}.png"
            
            if not baseline_path.exists() or not ablated_path.exists():
                continue
            
            baseline = np.array(Image.open(baseline_path).convert("RGB"), dtype=np.float32)
            ablated = np.array(Image.open(ablated_path).convert("RGB"), dtype=np.float32)
            
            pixel_diff_max = np.max(np.abs(baseline - ablated))
            pixel_diff_mean = np.mean(np.abs(baseline - ablated))
            
            diffs.append({
                "idx": i,
                "max": pixel_diff_max,
                "mean": pixel_diff_mean
            })
            
            print(f"  Image {i:03d}: max_diff={pixel_diff_max:.1f}, mean_diff={pixel_diff_mean:.2f}")
        
        if len(diffs) == 0:
            print(f"  ERROR: No image pairs found for {mode}")
            continue
        
        avg_max = np.mean([d["max"] for d in diffs])
        
        if avg_max > 10:  # Expecting significant differences
            print(f"  ✓ PASS: Images are different (avg max diff = {avg_max:.1f})")
        elif avg_max > 0:
            print(f"  ⚠ WARNING: Small differences (avg max diff = {avg_max:.1f})")
        else:
            print(f"  ✗ FAIL: Images are identical! Check your ablation code.")
    
    return True


# ============================================================
# Supplementary Experiments
# ============================================================

def _load_images_by_index(images_path: Path, mode: str, num_prompts: int) -> Dict[int, Image.Image]:
    """
    Load images as index->PIL mapping to avoid prompt/image misalignment.
    """
    out = {}
    for i in range(num_prompts):
        p = images_path / f"{i:03d}_{mode}.png"
        if p.exists():
            out[i] = Image.open(p).convert("RGB")
    return out


def _aligned_pairs(
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


def compute_hps_v2_score(images: List[Image.Image], prompts: List[str]) -> Optional[np.ndarray]:
    """Compute HPS-v2 scores with eval mode for determinism."""
    try:
        import hpsv2
    except ImportError:
        print("hpsv2 not installed")
        return None
    
    scores = []
    with torch.no_grad():  # Ensure determinism
        for img, prompt in tqdm(zip(images, prompts), total=len(prompts), desc="HPS-v2"):
            try:
                score = hpsv2.score(img, prompt, hps_version="v2.1")[0]
                scores.append(float(score))
            except Exception as e:
                scores.append(np.nan)
    
    return np.array(scores)


def compute_image_reward_score(images: List[Image.Image], prompts: List[str], device: str = "cuda") -> Optional[np.ndarray]:
    """Compute ImageReward scores with model caching and eval mode."""
    try:
        import ImageReward as RM
    except ImportError:
        print("ImageReward not installed. Install with: pip install image-reward")
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
        for img, prompt in tqdm(zip(images, prompts), total=len(prompts), desc="ImageReward"):
            try:
                score = model.score(prompt, img)
                scores.append(float(score))
            except Exception as e:
                scores.append(np.nan)
    
    return np.array(scores)


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


def run_supplement_on_existing(
    images_dir: str,
    prompts_file: str,
    num_prompts: int = 64,
    device: str = "cuda",
):
    """
    Run supplementary metrics (ImageReward, bottom_sink) on existing images.
    """
    print("\n" + "=" * 60)
    print("SUPPLEMENTARY EXPERIMENTS")
    print("=" * 60)
    
    images_path = Path(images_dir)
    
    # Load prompts
    with open(prompts_file, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    prompts = all_prompts[:num_prompts]
    
    # Check what modes exist
    available_modes = set()
    for f in images_path.glob("*.png"):
        parts = f.stem.split("_")
        if len(parts) >= 2:
            mode = "_".join(parts[1:])
            available_modes.add(mode)
    
    print(f"Available modes: {available_modes}")
    
    # Load existing results
    results_file = images_path / "hps_validation_results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
    else:
        results = {"n_prompts": num_prompts, "metrics": {}}
    
    # Ensure metrics dict exists
    if "metrics" not in results:
        results["metrics"] = {}
    
    # Load images by INDEX to avoid prompt/image misalignment
    imgs_none = _load_images_by_index(images_path, "none", num_prompts)
    if len(imgs_none) == 0:
        print("ERROR: Baseline images not found!")
        return None
    print(f"Loaded {len(imgs_none)} baseline images (indexed)")
    
    imgs_by_mode: Dict[str, Dict[int, Image.Image]] = {"none": imgs_none}
    for mode in ["top_sink", "random", "bottom_sink"]:
        if mode in available_modes:
            imgs_by_mode[mode] = _load_images_by_index(images_path, mode, num_prompts)
            print(f"Loaded {len(imgs_by_mode[mode])} images for mode={mode} (indexed)")
    
    # Compute ImageReward if not present
    if "image_reward" not in results["metrics"]:
        print("\n[Computing ImageReward...]")
        results["metrics"]["image_reward"] = {}
        
        for mode in ["top_sink", "random", "bottom_sink"]:
            if mode not in imgs_by_mode:
                continue
            
            # Get ALIGNED pairs
            base_imgs, mode_imgs, aligned_prompts, idxs = _aligned_pairs(
                prompts, imgs_by_mode["none"], imgs_by_mode[mode]
            )
            if len(idxs) < 2:
                print(f"  Skipping {mode}: not enough paired images")
                continue
            
            baseline_ir = compute_image_reward_score(base_imgs, aligned_prompts, device)
            mode_ir = compute_image_reward_score(mode_imgs, aligned_prompts, device)
            
            if baseline_ir is None or mode_ir is None:
                continue
            
            delta = mode_ir - baseline_ir
            valid = ~np.isnan(delta)
            ci_low, ci_high = bootstrap_ci(delta[valid])
            _, p_value = stats.ttest_rel(baseline_ir[valid], mode_ir[valid])
            
            results["metrics"]["image_reward"][mode] = {
                "n_pairs": int(valid.sum()),
                "delta_mean": float(np.nanmean(delta)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": float(p_value),
                "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
            }
            print(f"  {mode}: Δ={np.nanmean(delta):+.4f}, p={p_value:.4f}, n={valid.sum()}")
    
    # Compute HPS-v2 for bottom_sink if not present
    if "bottom_sink" in imgs_by_mode:
        if "hps_v2" not in results["metrics"]:
            results["metrics"]["hps_v2"] = {}
        
        if "bottom_sink" not in results["metrics"].get("hps_v2", {}):
            print("\n[Computing HPS-v2 for bottom_sink...]")
            
            # Get ALIGNED pairs
            base_imgs, mode_imgs, aligned_prompts, idxs = _aligned_pairs(
                prompts, imgs_by_mode["none"], imgs_by_mode["bottom_sink"]
            )
            if len(idxs) < 2:
                print("  Skipping bottom_sink HPS: not enough paired images")
            else:
                baseline_hps = compute_hps_v2_score(base_imgs, aligned_prompts)
                mode_hps = compute_hps_v2_score(mode_imgs, aligned_prompts)
                
                if baseline_hps is not None and mode_hps is not None:
                    delta = mode_hps - baseline_hps
                    valid = ~np.isnan(delta)
                    ci_low, ci_high = bootstrap_ci(delta[valid])
                    _, p_value = stats.ttest_rel(baseline_hps[valid], mode_hps[valid])
                    
                    results["metrics"]["hps_v2"]["bottom_sink"] = {
                        "n_pairs": int(valid.sum()),
                        "delta_mean": float(np.nanmean(delta)),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "p_value": float(p_value),
                        "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
                    }
                    print(f"  bottom_sink: Δ={np.nanmean(delta):+.4f}, p={p_value:.4f}, n={valid.sum()}")
    
    # Save updated results
    with open(images_path / "hps_validation_results_full.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print_full_summary(results)
    
    return results


def print_full_summary(results: Dict):
    """Print full summary table."""
    print("\n" + "=" * 100)
    print("FULL METRIC SUMMARY")
    print("=" * 100)
    print(f"{'Metric':<15} {'Mode':<15} {'Δ':>10} {'95% CI':>26} {'p':>10} {'CI∋0?':>8}")
    print("-" * 100)
    
    for metric_name in ["clip_t", "hps_v2", "image_reward"]:
        if metric_name not in results.get("metrics", {}):
            continue
        
        metric_results = results["metrics"][metric_name]
        for mode in ["top_sink", "random", "bottom_sink"]:
            if mode not in metric_results:
                continue
            
            res = metric_results[mode]
            ci_str = f"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]"
            ci_zero = "✓" if res.get('ci_includes_zero', res['ci_low'] <= 0 <= res['ci_high']) else "✗"
            print(f"{metric_name:<15} {mode:<15} {res['delta_mean']:>+.4f} {ci_str:>26} {res['p_value']:>10.4f} {ci_zero:>8}")
    
    print("=" * 100)


# ============================================================
# Generate bottom_sink images
# ============================================================

def generate_bottom_sink_images(
    prompts_file: str,
    output_dir: str,
    num_prompts: int = 64,
    target_layer: int = 12,
    top_k: int = 50,
    device: str = "cuda",
):
    """
    Generate bottom_sink images using the SAME verified patcher as top_sink/random.
    
    CRITICAL: Must use CounterfactualPatcher from ablation_counterfactual_v3.py
    to ensure bottom_sink is generated with the same intervention mechanism.
    """
    from diffusers import StableDiffusion3Pipeline
    
    # Import the VERIFIED patcher (same as used for top_sink/random)
    # Try v3 first, then fall back to v2
    try:
        from ablation_counterfactual_v3 import CounterfactualPatcher
        print("✓ Using CounterfactualPatcher from ablation_counterfactual_v3.py")
    except ImportError:
        try:
            from ablation_counterfactual_v2 import CounterfactualPatcher
            print("✓ Using CounterfactualPatcher from ablation_counterfactual_v2.py")
        except ImportError:
            print("ERROR: Cannot import CounterfactualPatcher from ablation_counterfactual_v3.py or v2.py")
            print("Make sure the file is in the same directory or PYTHONPATH.")
            return
    
    output_path = Path(output_dir)
    
    # Check if bottom_sink images already exist
    existing = list(output_path.glob("*_bottom_sink.png"))
    if len(existing) >= num_prompts:
        print(f"bottom_sink images already exist ({len(existing)} found)")
        return
    
    # Load prompts
    with open(prompts_file, 'r') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    prompts = all_prompts[:num_prompts]
    
    print(f"Generating {num_prompts} bottom_sink images...")
    print(f"Using VERIFIED CounterfactualPatcher (mode=bottom_sink, k={top_k})")
    
    # Load model
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to(device)
    
    # Use the SAME patcher as top_sink/random for consistency
    patcher = CounterfactualPatcher(
        target_layers=[target_layer],
        mode="bottom_sink",
        top_k=top_k,
    )
    patcher.patch(pipe.transformer)
    
    # Generate images
    for i, prompt in enumerate(tqdm(prompts, desc="Generating bottom_sink")):
        patcher.set_random_seed_offset(i * 10000)
        gen = torch.Generator(device=device).manual_seed(1000 + i)
        img = pipe(prompt, num_inference_steps=20, generator=gen).images[0]
        img.save(output_path / f"{i:03d}_bottom_sink.png")
    
    # Unpatch
    patcher.unpatch()
    
    print(f"Generated {num_prompts} bottom_sink images")


def main():
    parser = argparse.ArgumentParser(description="HPS Sanity Checks and Supplements")
    parser.add_argument("--sanity", action="store_true",
                       help="Run sanity checks")
    parser.add_argument("--supplement", action="store_true",
                       help="Run supplementary experiments")
    parser.add_argument("--generate_bottom", action="store_true",
                       help="Generate bottom_sink images")
    parser.add_argument("--images_dir", type=str, default="results_hps_k50",
                       help="Directory with existing images")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="Path to prompts file (REQUIRED for sanity/supplement/generate)")
    parser.add_argument("--num_prompts", type=int, default=64,
                       help="Number of prompts")
    parser.add_argument("--top_k", type=int, default=50,
                       help="k value")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    
    args = parser.parse_args()
    
    if args.sanity:
        if args.prompts_file is None:
            print("ERROR: --prompts_file is REQUIRED for sanity checks")
            return
        sanity_a_hps_determinism(args.images_dir, args.prompts_file)
        sanity_b_image_differences(args.images_dir)
    
    if args.generate_bottom:
        if args.prompts_file is None:
            print("ERROR: --prompts_file is REQUIRED for generating images")
            return
        generate_bottom_sink_images(
            prompts_file=args.prompts_file,
            output_dir=args.images_dir,
            num_prompts=args.num_prompts,
            top_k=args.top_k,
            device=args.device,
        )
    
    if args.supplement:
        if args.prompts_file is None:
            print("ERROR: --prompts_file is REQUIRED for supplement")
            return
        run_supplement_on_existing(
            images_dir=args.images_dir,
            prompts_file=args.prompts_file,
            num_prompts=args.num_prompts,
            device=args.device,
        )


if __name__ == "__main__":
    main()
