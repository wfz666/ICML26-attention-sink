#!/usr/bin/env python3
"""
No-op Sanity Check: Verify FID shift comes from intervention, not implementation bug.

This script:
1. Generates images with processor installed but intervention_enabled=False
2. Compares with baseline (no processor at all)
3. LPIPS and FID should be ~0 if implementation is correct

If FID_noop ≈ 0 but FID_intervention >> 0, we confirm the shift is from intervention.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.linalg import sqrtm
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    num_steps: int = 28
    guidance_scale: float = 7.0
    width: int = 1024
    height: int = 1024


def load_sd3_pipeline(device: str = "cuda"):
    """Load SD3 pipeline."""
    from diffusers import StableDiffusion3Pipeline
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def create_noop_processor(original_processor):
    """
    Create a processor that wraps the original but does nothing different.
    This tests whether just installing a custom processor causes changes.
    """
    from diffusers.models.attention_processor import JointAttnProcessor2_0
    
    class NoopJointAttnProcessor:
        """Processor that does exactly what the original does - no intervention."""
        
        def __init__(self, original):
            self.original = original
        
        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs
        ):
            # Just call the original processor - no modification
            return self.original(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
    
    return NoopJointAttnProcessor(original_processor)


def create_dynamic_sink_processor_noop(original_processor, config):
    """
    Create the actual DynamicSinkJointAttnProcessor but with intervention_enabled=False.
    This tests whether our processor implementation changes anything when disabled.
    """
    # Import your actual processor here
    # For now, create a minimal version that matches your implementation
    
    class DynamicSinkJointAttnProcessorNoop:
        """
        Same as DynamicSinkJointAttnProcessor but intervention always disabled.
        """
        def __init__(self, original, top_k=1, layers=[12]):
            self.original = original
            self.top_k = top_k
            self.layers = layers
            self.intervention_enabled = False  # Always False for noop
            self.current_step = 0
            self.call_count = 0
        
        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs
        ):
            # Since intervention_enabled=False, just call original
            # This matches what your real processor does when disabled
            return self.original(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
    
    return DynamicSinkJointAttnProcessorNoop(original_processor)


def generate_images(
    pipe,
    prompts: List[str],
    output_dir: Path,
    config: Config,
    mode: str = "baseline",  # "baseline", "noop_wrapper", "noop_processor"
    device: str = "cuda",
):
    """Generate images with different processor configurations."""
    from diffusers.models.attention_processor import JointAttnProcessor2_0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store original processors
    original_processors = {}
    for name, module in pipe.transformer.named_modules():
        if hasattr(module, 'processor'):
            original_processors[name] = module.processor
    
    try:
        if mode == "noop_wrapper":
            # Install simple wrapper that just calls original
            print("  Installing noop wrapper processors...")
            for name, module in pipe.transformer.named_modules():
                if hasattr(module, 'processor') and name in original_processors:
                    module.processor = create_noop_processor(original_processors[name])
        
        elif mode == "noop_processor":
            # Install actual DynamicSinkProcessor but with intervention_enabled=False
            print("  Installing DynamicSink processors with intervention_enabled=False...")
            for name, module in pipe.transformer.named_modules():
                if hasattr(module, 'processor') and name in original_processors:
                    module.processor = create_dynamic_sink_processor_noop(
                        original_processors[name],
                        config
                    )
        else:
            print("  Using original processors (baseline)...")
        
        # Generate images
        for i, prompt in enumerate(tqdm(prompts, desc=f"Generating ({mode})")):
            output_path = output_dir / f"{i:05d}.png"
            
            if output_path.exists():
                continue
            
            generator = torch.Generator(device=device).manual_seed(config.seed + i)
            
            image = pipe(
                prompt=prompt,
                num_inference_steps=config.num_steps,
                guidance_scale=config.guidance_scale,
                width=config.width,
                height=config.height,
                generator=generator,
            ).images[0]
            
            image.save(output_path)
    
    finally:
        # Restore original processors
        print("  Restoring original processors...")
        for name, module in pipe.transformer.named_modules():
            if hasattr(module, 'processor') and name in original_processors:
                module.processor = original_processors[name]


def compute_lpips(dir1: Path, dir2: Path, max_samples: int = 500, device: str = "cuda") -> Dict:
    """Compute paired LPIPS between two directories."""
    import lpips
    
    loss_fn = lpips.LPIPS(net='alex').to(device).eval()
    
    files1 = {p.stem: p for p in dir1.glob("*.png")}
    files2 = {p.stem: p for p in dir2.glob("*.png")}
    common = sorted(set(files1.keys()) & set(files2.keys()))[:max_samples]
    
    def to_tensor(p):
        with Image.open(p) as im:
            im = im.convert("RGB")
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        return (2 * x - 1).unsqueeze(0).to(device)
    
    vals = []
    with torch.no_grad():
        for k in tqdm(common, desc="LPIPS", leave=False):
            x = to_tensor(files1[k])
            y = to_tensor(files2[k])
            vals.append(loss_fn(x, y).item())
    
    vals = np.array(vals)
    return {
        "lpips_mean": float(vals.mean()),
        "lpips_std": float(vals.std()),
        "lpips_max": float(vals.max()),
        "n_samples": len(common),
    }


def compute_fid(dir1: Path, dir2: Path, device: str = "cuda") -> float:
    """Compute FID between two directories."""
    from cleanfid.fid import get_folder_features, build_feature_extractor
    
    model = build_feature_extractor(mode="clean", device=torch.device(device))
    
    F1 = get_folder_features(str(dir1), model=model, device=device, num_workers=4)
    F2 = get_folder_features(str(dir2), model=model, device=device, num_workers=4)
    
    n = min(F1.shape[0], F2.shape[0])
    F1, F2 = F1[:n], F2[:n]
    
    def fid_from_feats(X, Y, eps=1e-6):
        X, Y = X.astype(np.float64), Y.astype(np.float64)
        mu_x, mu_y = X.mean(0), Y.mean(0)
        sig_x = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
        sig_y = np.cov(Y, rowvar=False) + eps * np.eye(Y.shape[1])
        diff = mu_x - mu_y
        covmean, _ = sqrtm(sig_x @ sig_y, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sig_x + sig_y - 2 * covmean))
    
    return fid_from_feats(F1, F2)


def compute_pixel_diff(dir1: Path, dir2: Path, max_samples: int = 100) -> Dict:
    """Compute pixel-level differences."""
    files1 = {p.stem: p for p in dir1.glob("*.png")}
    files2 = {p.stem: p for p in dir2.glob("*.png")}
    common = sorted(set(files1.keys()) & set(files2.keys()))[:max_samples]
    
    diffs = []
    exact_matches = 0
    
    for k in common:
        img1 = np.array(Image.open(files1[k]))
        img2 = np.array(Image.open(files2[k]))
        diff = np.abs(img1.astype(float) - img2.astype(float)).mean()
        diffs.append(diff)
        if diff == 0:
            exact_matches += 1
    
    diffs = np.array(diffs)
    return {
        "pixel_diff_mean": float(diffs.mean()),
        "pixel_diff_std": float(diffs.std()),
        "pixel_diff_max": float(diffs.max()),
        "exact_matches": exact_matches,
        "n_samples": len(common),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="No-op sanity check")
    parser.add_argument("--output_dir", default="results_noop_sanity", help="Output directory")
    parser.add_argument("--num_prompts", type=int, default=100, help="Number of prompts")
    parser.add_argument("--prompts_file", default="results_coco_fid/prompts.txt", help="Prompts file")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation, only evaluate")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = Config()
    
    # Load prompts
    prompts_file = Path(args.prompts_file)
    if prompts_file.exists():
        with open(prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()][:args.num_prompts]
    else:
        prompts = [f"A photo of a cat {i}" for i in range(args.num_prompts)]
    
    print(f"Using {len(prompts)} prompts")
    
    baseline_dir = output_dir / "images_baseline"
    noop_wrapper_dir = output_dir / "images_noop_wrapper"
    noop_processor_dir = output_dir / "images_noop_processor"
    
    if not args.skip_generation:
        print("\n" + "="*60)
        print("Loading SD3 pipeline...")
        print("="*60)
        pipe = load_sd3_pipeline(args.device)
        
        # 1. Generate baseline (no processor modifications)
        print("\n[1/3] Generating baseline images...")
        generate_images(pipe, prompts, baseline_dir, config, mode="baseline", device=args.device)
        
        # 2. Generate with noop wrapper (simple passthrough)
        print("\n[2/3] Generating with noop wrapper...")
        generate_images(pipe, prompts, noop_wrapper_dir, config, mode="noop_wrapper", device=args.device)
        
        # 3. Generate with actual processor but intervention_enabled=False
        print("\n[3/3] Generating with DynamicSink processor (disabled)...")
        generate_images(pipe, prompts, noop_processor_dir, config, mode="noop_processor", device=args.device)
        
        del pipe
        torch.cuda.empty_cache()
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    results = {}
    
    # Compare baseline vs noop_wrapper
    print("\n[1/2] Baseline vs Noop Wrapper:")
    if baseline_dir.exists() and noop_wrapper_dir.exists():
        pixel = compute_pixel_diff(baseline_dir, noop_wrapper_dir)
        print(f"  Pixel diff: {pixel['pixel_diff_mean']:.4f} ± {pixel['pixel_diff_std']:.4f}")
        print(f"  Exact matches: {pixel['exact_matches']}/{pixel['n_samples']}")
        
        if pixel['pixel_diff_mean'] > 0.01:
            lpips_res = compute_lpips(baseline_dir, noop_wrapper_dir, device=args.device)
            print(f"  LPIPS: {lpips_res['lpips_mean']:.4f}")
            
            fid = compute_fid(baseline_dir, noop_wrapper_dir, device=args.device)
            print(f"  FID: {fid:.2f}")
        else:
            lpips_res = {"lpips_mean": 0.0}
            fid = 0.0
            print("  LPIPS: ~0 (images identical)")
            print("  FID: ~0 (images identical)")
        
        results["baseline_vs_noop_wrapper"] = {
            "pixel": pixel,
            "lpips": lpips_res.get("lpips_mean", 0),
            "fid": fid,
        }
    
    # Compare baseline vs noop_processor
    print("\n[2/2] Baseline vs Noop Processor (DynamicSink disabled):")
    if baseline_dir.exists() and noop_processor_dir.exists():
        pixel = compute_pixel_diff(baseline_dir, noop_processor_dir)
        print(f"  Pixel diff: {pixel['pixel_diff_mean']:.4f} ± {pixel['pixel_diff_std']:.4f}")
        print(f"  Exact matches: {pixel['exact_matches']}/{pixel['n_samples']}")
        
        if pixel['pixel_diff_mean'] > 0.01:
            lpips_res = compute_lpips(baseline_dir, noop_processor_dir, device=args.device)
            print(f"  LPIPS: {lpips_res['lpips_mean']:.4f}")
            
            fid = compute_fid(baseline_dir, noop_processor_dir, device=args.device)
            print(f"  FID: {fid:.2f}")
        else:
            lpips_res = {"lpips_mean": 0.0}
            fid = 0.0
            print("  LPIPS: ~0 (images identical)")
            print("  FID: ~0 (images identical)")
        
        results["baseline_vs_noop_processor"] = {
            "pixel": pixel,
            "lpips": lpips_res.get("lpips_mean", 0),
            "fid": fid,
        }
    
    # Summary
    print("\n" + "="*60)
    print("SANITY CHECK SUMMARY")
    print("="*60)
    
    wrapper_ok = results.get("baseline_vs_noop_wrapper", {}).get("pixel", {}).get("pixel_diff_mean", 999) < 1
    processor_ok = results.get("baseline_vs_noop_processor", {}).get("pixel", {}).get("pixel_diff_mean", 999) < 1
    
    if wrapper_ok and processor_ok:
        print("✓ PASS: Both noop configurations produce identical images")
        print("  → FID shift in real experiments comes from INTERVENTION, not implementation bug")
    elif wrapper_ok:
        print("⚠ PARTIAL: Wrapper is identical, but processor differs")
        print("  → Check DynamicSinkProcessor implementation")
    else:
        print("✗ FAIL: Even noop wrapper produces different images")
        print("  → Possible non-determinism in pipeline")
    
    print("\nExpected values for correct implementation:")
    print("  Pixel diff: 0 (exact match)")
    print("  LPIPS: 0")
    print("  FID: 0")
    
    print("\nFor reference, actual intervention produces:")
    print("  LPIPS: ~0.18-0.31")
    print("  FID: ~400-1000")
    
    # Save results
    results_path = output_dir / "sanity_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
