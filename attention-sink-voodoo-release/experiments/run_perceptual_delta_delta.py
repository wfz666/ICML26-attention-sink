#!/usr/bin/env python3
"""
Rebuttal P1: Perceptual Sink-Specificity (ΔΔ LPIPS)
=====================================================

Directly addresses Reviewer G5q6's question:
  "Is the perceptual shift sink-specific, or does masking any tokens
   produce comparable LPIPS/FID?"

And strengthens responses to ZxoJ, uCqt, ugYV who all ask about
the nature of perceptual drift.

Design:
  - 3 modes:  none (baseline), top_sink, random
  - 2 budgets: k=1 (standard), k=5 (stronger)
  - N=64 prompts, layer 12, seed-paired
  - Saves ALL images (for qualitative panel / FID later)
  - Computes: CLIP-T, LPIPS, and ΔΔ = Δ_sink - Δ_random

Key output: ΔΔ_LPIPS table showing whether perceptual drift is
sink-specific or a generic masking artifact.

Usage:
    # Standard (1 GPU, ~40 min)
    python rebuttal_perceptual_delta_delta.py \\
        --prompts_file generation_prompts.txt \\
        --output_dir results_rebuttal_p1

    # With HPS-v2 (slower but more complete)
    python rebuttal_perceptual_delta_delta.py \\
        --prompts_file generation_prompts.txt \\
        --output_dir results_rebuttal_p1 \\
        --compute_hps
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F_torch
from PIL import Image
from tqdm import tqdm
from scipy import stats
from scipy.stats import wilcoxon

# Import from existing codebase
sys.path.insert(0, str(Path(__file__).parent))
from ablation_counterfactual_v3 import (
    CounterfactualPatcher,
    run_sanity_checks,
    compute_clip_score,
    bootstrap_ci_seeded,
    holm_bonferroni_correction,
)


# ============================================================
# LPIPS computation
# ============================================================

_lpips_model = None

def get_lpips_model(device: str):
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    return _lpips_model


def compute_lpips_paired(
    images_a: List[Image.Image],
    images_b: List[Image.Image],
    device: str,
) -> np.ndarray:
    """Compute per-pair LPIPS distances."""
    model = get_lpips_model(device)
    scores = []
    for img_a, img_b in zip(images_a, images_b):
        ta = torch.from_numpy(np.array(img_a.convert("RGB"))).permute(2, 0, 1).float() / 127.5 - 1
        tb = torch.from_numpy(np.array(img_b.convert("RGB"))).permute(2, 0, 1).float() / 127.5 - 1
        ta = ta.unsqueeze(0).to(device)
        tb = tb.unsqueeze(0).to(device)
        with torch.no_grad():
            d = model(ta, tb).item()
        scores.append(d)
    return np.array(scores)


# ============================================================
# ΔΔ computation (consistent with compute_delta_delta.py)
# ============================================================

def compute_delta_delta(
    baseline: np.ndarray,
    top_sink: np.ndarray,
    random: np.ndarray,
    metric_name: str = "metric",
    seed: int = 42,
) -> Dict:
    """
    ΔΔ = (top_sink - baseline) - (random - baseline) = top_sink - random

    One-sided test: ΔΔ > 0 means sink masking perturbs MORE than random.
    For LPIPS (higher = more different), sink-specificity = ΔΔ > 0.
    For CLIP-T (higher = better), sink-specificity of damage = ΔΔ < 0.
    """
    d = (top_sink - baseline) - (random - baseline)  # = top_sink - random
    valid = ~np.isnan(d)
    d_valid = d[valid]

    if len(d_valid) < 2:
        return {"metric": metric_name, "n": 0, "dd_mean": np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
                "p_two_sided": np.nan, "ci_includes_zero": True,
                "interpretation": "insufficient data"}

    dd_mean = float(d_valid.mean())
    ci_low, ci_high = bootstrap_ci_seeded(d_valid, seed=seed)

    # Two-sided t-test
    _, p_t = stats.ttest_1samp(d_valid, 0.0)

    # Wilcoxon (non-parametric)
    try:
        p_w = wilcoxon(d_valid).pvalue
    except Exception:
        p_w = np.nan

    ci_zero = bool(ci_low <= 0 <= ci_high)

    # Interpretation depends on metric direction
    if metric_name.lower() == "lpips":
        # LPIPS: higher = more perturbed. ΔΔ > 0 means sink-specific drift.
        if dd_mean > 0 and not ci_zero:
            interp = "SINK-SPECIFIC (sinks perturb more than random)"
        elif dd_mean < 0 and not ci_zero:
            interp = "RANDOM-SPECIFIC (random perturbs more than sinks)"
        else:
            interp = "NO DIFFERENCE (sink and random produce similar drift)"
    else:
        # CLIP-T / HPS: lower = worse. ΔΔ < 0 means sink-specific damage.
        if dd_mean < 0 and not ci_zero:
            interp = "SINK-SPECIFIC (sinks degrade more than random)"
        elif dd_mean > 0 and not ci_zero:
            interp = "RANDOM-SPECIFIC (random degrades more than sinks)"
        else:
            interp = "NO DIFFERENCE"

    return {
        "metric": metric_name,
        "n": int(len(d_valid)),
        "dd_mean": float(dd_mean),
        "dd_std": float(d_valid.std()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_ttest": float(p_t),
        "p_wilcoxon": float(p_w),
        "ci_includes_zero": ci_zero,
        "interpretation": interp,
    }


# ============================================================
# Image generation for one k-budget
# ============================================================

def generate_all_modes(
    pipe,
    prompts: List[str],
    target_layer: int,
    top_k: int,
    device: str,
    output_path: Path,
) -> Dict[str, List[Image.Image]]:
    """
    Generate images for none/top_sink/random at given k.
    Saves ALL images to disk. Returns dict of image lists.
    """
    modes = ["none", "top_sink", "random"]
    all_images = {m: [] for m in modes}

    for mode in modes:
        img_dir = output_path / f"images_k{top_k}_{mode}"
        img_dir.mkdir(parents=True, exist_ok=True)

        patcher = CounterfactualPatcher(
            target_layers=[target_layer],
            mode=mode,
            top_k=top_k,
        )
        patcher.patch(pipe.transformer)

        for i, prompt in enumerate(tqdm(prompts, desc=f"k={top_k}/{mode}")):
            patcher.set_random_seed_offset(i * 10000)
            gen = torch.Generator(device=device).manual_seed(1000 + i)
            img = pipe(prompt, num_inference_steps=20, generator=gen).images[0]
            all_images[mode].append(img)

            # Save every image
            img.save(img_dir / f"{i:04d}.png")

        patcher.unpatch()

    return all_images


# ============================================================
# Evaluate one k-budget
# ============================================================

def evaluate_budget(
    all_images: Dict[str, List[Image.Image]],
    prompts: List[str],
    top_k: int,
    device: str,
    compute_hps: bool = False,
) -> Dict:
    """Compute all metrics and ΔΔ for one k-budget."""

    baseline_imgs = all_images["none"]
    sink_imgs = all_images["top_sink"]
    random_imgs = all_images["random"]
    n = len(prompts)

    results = {"k": top_k, "n": n}

    # --- CLIP-T ---
    print(f"  [k={top_k}] Computing CLIP-T ...")
    clip_base = compute_clip_score(baseline_imgs, prompts, device)
    clip_sink = compute_clip_score(sink_imgs, prompts, device)
    clip_rand = compute_clip_score(random_imgs, prompts, device)

    delta_sink_clip = clip_sink - clip_base
    delta_rand_clip = clip_rand - clip_base

    ci_sink = bootstrap_ci_seeded(delta_sink_clip, seed=42)
    ci_rand = bootstrap_ci_seeded(delta_rand_clip, seed=43)
    _, p_sink_clip = stats.ttest_rel(clip_base, clip_sink)
    _, p_rand_clip = stats.ttest_rel(clip_base, clip_rand)

    results["clip_t"] = {
        "top_sink": {
            "delta_mean": float(delta_sink_clip.mean()),
            "ci": [float(ci_sink[0]), float(ci_sink[1])],
            "p": float(p_sink_clip),
        },
        "random": {
            "delta_mean": float(delta_rand_clip.mean()),
            "ci": [float(ci_rand[0]), float(ci_rand[1])],
            "p": float(p_rand_clip),
        },
        "delta_delta": compute_delta_delta(clip_base, clip_sink, clip_rand, "CLIP-T"),
    }

    # --- LPIPS ---
    print(f"  [k={top_k}] Computing LPIPS ...")
    lpips_sink = compute_lpips_paired(baseline_imgs, sink_imgs, device)
    lpips_rand = compute_lpips_paired(baseline_imgs, random_imgs, device)

    results["lpips"] = {
        "top_sink": {
            "mean": float(lpips_sink.mean()),
            "std": float(lpips_sink.std()),
        },
        "random": {
            "mean": float(lpips_rand.mean()),
            "std": float(lpips_rand.std()),
        },
        "delta_delta": compute_delta_delta(
            np.zeros(n),  # LPIPS is already a distance, not a score;
            lpips_sink,    # ΔΔ = lpips_sink - lpips_rand
            lpips_rand,
            "LPIPS",
        ),
    }

    # --- HPS-v2 (optional) ---
    if compute_hps:
        print(f"  [k={top_k}] Computing HPS-v2 ...")
        try:
            from ablation_counterfactual_v3 import compute_hps_v2_score
            hps_base = compute_hps_v2_score(baseline_imgs, prompts, device)
            hps_sink = compute_hps_v2_score(sink_imgs, prompts, device)
            hps_rand = compute_hps_v2_score(random_imgs, prompts, device)

            if hps_base is not None and hps_sink is not None and hps_rand is not None:
                delta_sink_hps = hps_sink - hps_base
                delta_rand_hps = hps_rand - hps_base
                ci_s = bootstrap_ci_seeded(delta_sink_hps, seed=44)
                ci_r = bootstrap_ci_seeded(delta_rand_hps, seed=45)
                _, p_s = stats.ttest_rel(hps_base, hps_sink)
                _, p_r = stats.ttest_rel(hps_base, hps_rand)

                results["hps_v2"] = {
                    "top_sink": {
                        "delta_mean": float(np.nanmean(delta_sink_hps)),
                        "ci": [float(ci_s[0]), float(ci_s[1])],
                        "p": float(p_s),
                    },
                    "random": {
                        "delta_mean": float(np.nanmean(delta_rand_hps)),
                        "ci": [float(ci_r[0]), float(ci_r[1])],
                        "p": float(p_r),
                    },
                    "delta_delta": compute_delta_delta(
                        hps_base, hps_sink, hps_rand, "HPS-v2"
                    ),
                }
        except Exception as e:
            print(f"  HPS-v2 failed: {e}")

    # --- Raw scores for downstream use ---
    results["raw_scores"] = {
        "clip_base": clip_base.tolist(),
        "clip_sink": clip_sink.tolist(),
        "clip_rand": clip_rand.tolist(),
        "lpips_sink": lpips_sink.tolist(),
        "lpips_rand": lpips_rand.tolist(),
    }

    return results


# ============================================================
# Pretty printing
# ============================================================

def print_rebuttal_table(all_results: Dict[int, Dict]):
    """Print rebuttal-ready summary."""

    print("\n" + "=" * 90)
    print("REBUTTAL P1: PERCEPTUAL SINK-SPECIFICITY (ΔΔ)")
    print("=" * 90)

    # --- Per-mode Δ table ---
    print(f"\n{'':─<90}")
    print(f"{'k':>3}  {'Mode':<12} {'ΔCLIP-T':>10} {'95% CI':>24} "
          f"{'LPIPS':>8} {'LPIPS std':>10}")
    print(f"{'':─<90}")

    for k, res in sorted(all_results.items()):
        for mode in ["top_sink", "random"]:
            clip_d = res["clip_t"][mode]
            lpips_d = res["lpips"][mode]
            ci = f"[{clip_d['ci'][0]:+.4f}, {clip_d['ci'][1]:+.4f}]"
            print(f"{k:>3}  {mode:<12} {clip_d['delta_mean']:>+10.4f} {ci:>24} "
                  f"{lpips_d['mean']:>8.4f} {lpips_d['std']:>10.4f}")
        print()

    # --- ΔΔ table (the key result) ---
    print(f"{'':─<90}")
    print(f"{'k':>3}  {'Metric':<10} {'ΔΔ mean':>10} {'95% CI':>24} "
          f"{'p(t)':>8} {'p(W)':>8} {'CI∋0':>5}  Interpretation")
    print(f"{'':─<90}")

    for k, res in sorted(all_results.items()):
        for metric_key in ["clip_t", "lpips"]:
            dd = res[metric_key]["delta_delta"]
            ci = f"[{dd['ci_low']:+.4f}, {dd['ci_high']:+.4f}]"
            ci0 = "✓" if dd["ci_includes_zero"] else "✗"
            pt = f"{dd['p_ttest']:.4f}" if not np.isnan(dd["p_ttest"]) else "  nan"
            pw = f"{dd['p_wilcoxon']:.4f}" if not np.isnan(dd["p_wilcoxon"]) else "  nan"
            print(f"{k:>3}  {dd['metric']:<10} {dd['dd_mean']:>+10.4f} {ci:>24} "
                  f"{pt:>8} {pw:>8} {ci0:>5}  {dd['interpretation']}")

        # HPS-v2 if available
        if "hps_v2" in res:
            dd = res["hps_v2"]["delta_delta"]
            ci = f"[{dd['ci_low']:+.4f}, {dd['ci_high']:+.4f}]"
            ci0 = "✓" if dd["ci_includes_zero"] else "✗"
            pt = f"{dd['p_ttest']:.4f}" if not np.isnan(dd["p_ttest"]) else "  nan"
            pw = f"{dd['p_wilcoxon']:.4f}" if not np.isnan(dd["p_wilcoxon"]) else "  nan"
            print(f"{k:>3}  {dd['metric']:<10} {dd['dd_mean']:>+10.4f} {ci:>24} "
                  f"{pt:>8} {pw:>8} {ci0:>5}  {dd['interpretation']}")

        print()

    print("=" * 90)
    print("ΔΔ = Δ_sink − Δ_random.  For LPIPS: ΔΔ>0 = sink perturbs MORE.")
    print("                         For CLIP-T/HPS: ΔΔ<0 = sink damages MORE.")
    print("=" * 90)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rebuttal P1: Perceptual sink-specificity via ΔΔ LPIPS"
    )
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_rebuttal_p1")
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--target_layer", type=int, default=12)
    parser.add_argument("--k_values", type=str, default="1,5",
                        help="Comma-separated k values to sweep")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute_hps", action="store_true",
                        help="Also compute HPS-v2 (slower)")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load prompts
    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    prompts = prompts[:args.num_prompts]
    print(f"Loaded {len(prompts)} prompts")

    # Save prompts
    with open(output_path / "prompts.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")

    # Load SD3
    print("\nLoading SD3 ...")
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    print("SD3 loaded.\n")

    # Sanity checks (once, at k=1)
    print("[Sanity Checks]")
    sanity_ok, patched_names = run_sanity_checks(
        pipe, prompts, args.device,
        target_layer=args.target_layer, top_k=1,
    )
    if not sanity_ok:
        print("Sanity check failed. Aborting.")
        return

    # Run for each k
    all_results = {}

    for k in k_values:
        print(f"\n{'=' * 70}")
        print(f"  Budget k={k}  (layer {args.target_layer}, N={len(prompts)})")
        print(f"{'=' * 70}")

        # Generate
        all_images = generate_all_modes(
            pipe, prompts, args.target_layer, k, args.device, output_path,
        )

        # Evaluate
        results = evaluate_budget(
            all_images, prompts, k, args.device,
            compute_hps=args.compute_hps,
        )
        all_results[k] = results

        # Save per-k results
        with open(output_path / f"results_k{k}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Free images from memory (they're on disk)
        del all_images
        torch.cuda.empty_cache()

    # Save combined results (without raw scores to keep file small)
    combined = {}
    for k, res in all_results.items():
        combined[k] = {key: val for key, val in res.items() if key != "raw_scores"}
    with open(output_path / "results_combined.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    # Print summary
    print_rebuttal_table(all_results)

    # --- Suggested rebuttal language ---
    print("\n" + "-" * 70)
    print("SUGGESTED REBUTTAL LANGUAGE")
    print("-" * 70)

    k1 = all_results.get(1)
    k5 = all_results.get(5, all_results.get(k_values[-1]))

    if k1:
        dd_lpips_k1 = k1["lpips"]["delta_delta"]
        dd_clip_k1 = k1["clip_t"]["delta_delta"]
        print(f"""
At k=1 (standard setting):
  LPIPS ΔΔ = {dd_lpips_k1['dd_mean']:+.4f}, 95% CI [{dd_lpips_k1['ci_low']:+.4f}, {dd_lpips_k1['ci_high']:+.4f}]
  CLIP-T ΔΔ = {dd_clip_k1['dd_mean']:+.4f}, 95% CI [{dd_clip_k1['ci_low']:+.4f}, {dd_clip_k1['ci_high']:+.4f}]
  → {dd_lpips_k1['interpretation']}""")

    if k5 and k5["k"] != 1:
        dd_lpips_k5 = k5["lpips"]["delta_delta"]
        dd_clip_k5 = k5["clip_t"]["delta_delta"]
        print(f"""
At k={k5['k']} (stronger):
  LPIPS ΔΔ = {dd_lpips_k5['dd_mean']:+.4f}, 95% CI [{dd_lpips_k5['ci_low']:+.4f}, {dd_lpips_k5['ci_high']:+.4f}]
  CLIP-T ΔΔ = {dd_clip_k5['dd_mean']:+.4f}, 95% CI [{dd_clip_k5['ci_low']:+.4f}, {dd_clip_k5['ci_high']:+.4f}]
  → {dd_lpips_k5['interpretation']}""")

    print(f"\nAll images saved under: {output_path}/images_k*_*/")
    print(f"Results: {output_path}/results_combined.json")


if __name__ == "__main__":
    main()
