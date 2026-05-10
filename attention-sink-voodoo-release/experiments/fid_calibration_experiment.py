#!/usr/bin/env python3
"""
FID Calibration Baseline

目标：给 FID shift 提供参照系，消除 reviewer 对数值的疑虑

生成多组对比：
1. Baseline (seed A) vs Baseline (seed B) - 随机种子的 FID 差异
2. Baseline vs CFG ±1 - CFG 变化的影响
3. Baseline vs fewer steps - 步数变化的影响
4. Baseline vs Intervention - 我们的干预

这样可以说明：
"The observed FID shift is comparable to X but smaller/larger than Y."
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import shutil


def generate_images(
    pipe,
    prompts: List[str],
    output_dir: Path,
    seeds: List[int],
    num_steps: int = 20,
    cfg: float = 7.5,
    device: str = "cuda",
    desc: str = "Generating",
):
    """Generate images with given settings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (prompt, seed) in enumerate(tqdm(zip(prompts, seeds), total=len(prompts), desc=desc)):
        gen = torch.Generator(device=device).manual_seed(seed)
        img = pipe(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=cfg,
            generator=gen,
        ).images[0]
        # Ensure consistent mode for pytorch-fid (avoid RGBA/P/alpha surprises)
        img = img.convert("RGB")
        img.save(output_dir / f"{i:04d}.png")


def compute_fid(dir1: Path, dir2: Path, device: str = "cuda") -> float:
    """Compute FID between two image directories."""
    from pytorch_fid import fid_score
    
    fid = fid_score.calculate_fid_given_paths(
        [str(dir1), str(dir2)],
        batch_size=50,
        device=device,
        dims=2048,
    )
    return fid


def run_fid_calibration(
    prompts_file: str,
    output_dir: str,
    num_prompts: int = 100,
    model: str = "sd3",  # "sd3" or "sdxl"
    device: str = "cuda",
):
    """Run FID calibration experiment."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    prompts = prompts[:num_prompts]
    print(f"Using {len(prompts)} prompts")
    
    # Load model
    print(f"\nLoading {model.upper()}...")
    if model == "sd3":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
        )
    else:  # sdxl
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        )
    
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Define configurations
    configs = {
        # Reference baseline
        "baseline_A": {"seeds": list(range(1000, 1000 + num_prompts)), "steps": 20, "cfg": 7.5, "scheduler": "default"},
        
        # Different seeds (same settings)
        "baseline_B": {"seeds": list(range(2000, 2000 + num_prompts)), "steps": 20, "cfg": 7.5, "scheduler": "default"},
        
        # CFG variations
        "cfg_6.5": {"seeds": list(range(1000, 1000 + num_prompts)), "steps": 20, "cfg": 6.5, "scheduler": "default"},
        "cfg_8.5": {"seeds": list(range(1000, 1000 + num_prompts)), "steps": 20, "cfg": 8.5, "scheduler": "default"},
        
        # Step variations
        "steps_15": {"seeds": list(range(1000, 1000 + num_prompts)), "steps": 15, "cfg": 7.5, "scheduler": "default"},
        "steps_10": {"seeds": list(range(1000, 1000 + num_prompts)), "steps": 10, "cfg": 7.5, "scheduler": "default"},
        
        # Scheduler variation (Euler)
        "euler": {"seeds": list(range(1000, 1000 + num_prompts)), "steps": 20, "cfg": 7.5, "scheduler": "euler"},
    }
    
    # Generate all configurations
    print("\n[1/2] Generating images for each configuration...")
    
    # Save default scheduler
    default_scheduler = pipe.scheduler
    skipped_configs = []
    
    for config_name, config in configs.items():
        scheduler_name = config.get("scheduler", "default")
        print(f"\n  {config_name}: steps={config['steps']}, cfg={config['cfg']}, scheduler={scheduler_name}")
        config_dir = output_path / "images" / config_name
        
        # Switch scheduler if needed
        if scheduler_name == "euler":
            from diffusers import EulerDiscreteScheduler
            try:
                pipe.scheduler = EulerDiscreteScheduler.from_config(default_scheduler.config)
            except Exception as e:
                print(f"    [WARN] Euler scheduler not compatible with this pipeline: {e}")
                print("    Skipping Euler config for this model.")
                skipped_configs.append(config_name)
                continue
        else:
            pipe.scheduler = default_scheduler
        
        generate_images(
            pipe=pipe,
            prompts=prompts,
            output_dir=config_dir,
            seeds=config["seeds"],
            num_steps=config["steps"],
            cfg=config["cfg"],
            device=device,
            desc=f"  {config_name}",
        )
    
    # Restore default scheduler
    pipe.scheduler = default_scheduler
    
    # Compute FID for all pairs
    print("\n[2/2] Computing FID for each comparison...")
    
    baseline_dir = output_path / "images" / "baseline_A"
    
    comparisons = [
        ("baseline_A", "baseline_B", "Seed variation (same settings)"),
        ("baseline_A", "cfg_6.5", "CFG 7.5 → 6.5 (Δ = -1)"),
        ("baseline_A", "cfg_8.5", "CFG 7.5 → 8.5 (Δ = +1)"),
        ("baseline_A", "steps_15", "Steps 20 → 15 (Δ = -5)"),
        ("baseline_A", "steps_10", "Steps 20 → 10 (Δ = -10)"),
        ("baseline_A", "euler", "Scheduler: default → Euler"),
    ]
    
    fid_results = []
    
    for dir1_name, dir2_name, description in comparisons:
        # Skip if config was skipped due to incompatibility
        if dir2_name in skipped_configs:
            print(f"\n  Skipping FID: {dir1_name} vs {dir2_name} (config was skipped)")
            continue
        
        dir1 = output_path / "images" / dir1_name
        dir2 = output_path / "images" / dir2_name
        
        # Check if directories exist
        if not dir1.exists() or not dir2.exists():
            print(f"\n  Skipping FID: {dir1_name} vs {dir2_name} (directory missing)")
            continue
        
        print(f"\n  Computing FID: {dir1_name} vs {dir2_name}...")
        fid = compute_fid(dir1, dir2, device=device)
        
        fid_results.append({
            "comparison": f"{dir1_name} vs {dir2_name}",
            "description": description,
            "fid": float(fid),
        })
        
        print(f"    FID = {fid:.2f}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("FID CALIBRATION RESULTS")
    print("=" * 80)
    print(f"Model: {model.upper()}")
    print(f"N = {num_prompts} prompts")
    print("-" * 80)
    print(f"{'Comparison':<40} {'Description':<30} {'FID':>8}")
    print("-" * 80)
    
    for r in fid_results:
        print(f"{r['comparison']:<40} {r['description']:<30} {r['fid']:>8.2f}")
    
    print("=" * 80)
    
    # Save results
    results = {
        "model": model,
        "num_prompts": num_prompts,
        "comparisons": fid_results,
        "note": "Use these as calibration baselines for interpreting intervention FID shifts",
    }
    
    with open(output_path / "fid_calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path / 'fid_calibration_results.json'}")
    
    # Generate LaTeX table
    latex = generate_latex_table(fid_results, model, num_prompts)
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    print(latex)
    
    with open(output_path / "fid_calibration_table.tex", "w") as f:
        f.write(latex)
    
    return results


def generate_latex_table(fid_results: List[Dict], model: str, num_prompts: int) -> str:
    """Generate LaTeX table for FID calibration."""
    
    latex = r"""\begin{table}[t]
\centering
\caption{\textbf{FID calibration baselines.}
FID shifts from common hyperparameter variations provide reference points
for interpreting intervention effects. Seed variation establishes the
stochastic baseline; CFG and step changes show sensitivity to generation settings.}
\label{tab:fid_calibration}
\vskip 0.1in
\begin{small}
\begin{tabular}{lc}
\toprule
Comparison & FID \\
\midrule
"""
    
    for r in fid_results:
        desc = r['description'].replace("→", "$\\rightarrow$").replace("Δ", "$\\Delta$")
        latex += f"{desc} & {r['fid']:.1f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{small}
\end{table}
"""
    
    return latex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_fid_calibration")
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--model", type=str, default="sd3", choices=["sd3", "sdxl"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    run_fid_calibration(
        prompts_file=args.prompts_file,
        output_dir=args.output_dir,
        num_prompts=args.num_prompts,
        model=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
