#!/usr/bin/env python
"""
多层干预实验
============
同时在多个层（6+12+18）进行干预，验证"单层不重要，多层是否重要"

Usage:
    python run_multilayer.py --model sd3 --num_samples 32
"""

import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Dict
import matplotlib.pyplot as plt

# 复用现有组件
import sys
sys.path.insert(0, '.')

from sink_analysis import (
    ExperimentConfig, 
    InterventionConfig, 
    TransformerPatcher,
)


def run_multilayer_experiment(
    model_name: str = "sd3",
    intervention_layers: List[int] = [6, 12, 18],
    num_samples: int = 32,
    num_steps: int = 50,
    seed: int = 42,
    prompts_file: str = None,
    output_dir: str = "./results_multilayer",
    device: str = "cuda"
):
    """
    多层干预实验
    
    比较：
    1. baseline (无干预)
    2. 单层干预 (layer 12)
    3. 多层干预 (layer 6+12+18)
    """
    from diffusers import StableDiffusion3Pipeline
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"Loading {model_name} pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    ).to(device)
    
    # 加载prompts
    if prompts_file and Path(prompts_file).exists():
        with open(prompts_file) as f:
            prompts = [l.strip() for l in f if l.strip()][:num_samples]
        print(f"Loaded {len(prompts)} prompts from {prompts_file}")
    else:
        prompts = [
            "A majestic mountain landscape at sunset with snow-capped peaks",
            "A cozy coffee shop interior with warm lighting",
            "A futuristic cityscape with flying cars and neon lights",
            "A serene Japanese garden with cherry blossoms",
            "A portrait of a wise old wizard with a long white beard",
            "An underwater scene with colorful coral reefs",
            "A steampunk mechanical owl with brass gears",
            "A mystical forest with bioluminescent mushrooms",
        ]
        # 扩展到所需数量
        while len(prompts) < num_samples:
            prompts.extend(prompts)
        prompts = prompts[:num_samples]
        print(f"Using {len(prompts)} default prompts")
    
    # 保存prompts
    with open(output_path / "prompts.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    
    # 定义实验条件
    conditions = {
        "baseline": {
            "intervention_type": "none",
            "intervention_layers": [],
        },
        "single_layer_12": {
            "intervention_type": "both",
            "score_scale": 0.0,
            "value_method": "zero",
            "intervention_layers": [12],
        },
        "multi_layer_6_12_18": {
            "intervention_type": "both",
            "score_scale": 0.0,
            "value_method": "zero",
            "intervention_layers": [6, 12, 18],
        },
    }
    
    results = {}
    
    for cond_name, cond_params in conditions.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}")
        print(f"Intervention layers: {cond_params.get('intervention_layers', [])}")
        print("="*60)
        
        # 配置
        intervention = InterventionConfig(
            intervention_type=cond_params["intervention_type"],
            score_method="prob_scale",
            score_scale=cond_params.get("score_scale", 1.0),
            value_method=cond_params.get("value_method", "none"),
            intervention_layers=cond_params.get("intervention_layers", []),
            sink_token_indices=[0],
        )
        
        config = ExperimentConfig(
            model_name=model_name,
            num_steps=num_steps,
            num_samples=num_samples,
            seed=seed,
            measure_layers=[6, 12, 18],
            intervention=intervention,
        )
        
        # Patch并打印layer mapping用于验证
        patcher = TransformerPatcher(config)
        patcher.patch(pipe.transformer)
        
        # 打印layer mapping（只在第一个condition时打印）
        if cond_name == list(conditions.keys())[0]:
            print("\n  Layer index mapping (processor_name -> idx):")
            for name, proc in list(patcher.processors.items())[:5]:
                print(f"    {name}: layer_idx={proc.layer_idx}")
            print("    ...")
        
        images = []
        
        try:
            # 只在每个 condition 开始时清空一次，累计所有 prompts 的 metrics
            patcher.clear_metrics()
            
            for prompt_idx, prompt in enumerate(tqdm(prompts, desc=cond_name)):
                # 关键：per-prompt配对噪声，确保不同condition之间严格配对
                generator = torch.Generator(device=device).manual_seed(seed + prompt_idx)
                
                def step_callback(pipe, step, timestep, callback_kwargs):
                    patcher.set_timestep(step / (num_steps - 1) if num_steps > 1 else 0)
                    return callback_kwargs
                
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    generator=generator,
                    callback_on_step_end=step_callback,
                    output_type="pil"
                )
                images.append(result.images[0])
        
        finally:
            patcher.unpatch()
        
        # 保存图片
        img_dir = output_path / f"images_{cond_name}"
        img_dir.mkdir(exist_ok=True)
        for idx, img in enumerate(images):
            img.save(img_dir / f"{idx:03d}.png")
        
        # 获取sink metrics
        df = patcher.get_metrics_dataframe()
        if len(df) > 0:
            mean_sink = df['sink_ratio'].mean()
            print(f"  Mean sink ratio: {mean_sink:.4f}")
            results[cond_name] = {
                "mean_sink_ratio": float(mean_sink),
                "num_images": len(images),
            }
        else:
            results[cond_name] = {"num_images": len(images)}
    
    # 保存配置
    config_data = {
        "model": model_name,
        "num_samples": num_samples,
        "num_steps": num_steps,
        "seed": seed,
        "intervention_layers": intervention_layers,
        "conditions": list(conditions.keys()),
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Multilayer experiment completed!")
    print(f"Results saved to: {output_path}")
    print("="*60)
    print("\nNext steps:")
    print(f"  1. Evaluate CLIP: python eval_paired_delta.py {output_path} multilayer")
    print(f"  2. Evaluate ImageReward: python eval_imagereward.py {output_path} multilayer")
    
    return results


def evaluate_multilayer(results_dir: str):
    """评估多层干预实验的质量"""
    from transformers import CLIPProcessor, CLIPModel
    
    results_path = Path(results_dir)
    
    # 加载prompts
    with open(results_path / "prompts.txt") as f:
        prompts = [l.strip() for l in f if l.strip()]
    
    # 加载CLIP
    print("Loading CLIP...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    
    # 评估每个条件
    all_scores = {}
    
    for img_dir in sorted(results_path.glob("images_*")):
        cond_name = img_dir.name.replace("images_", "")
        print(f"\nEvaluating: {cond_name}")
        
        img_files = sorted(img_dir.glob("*.png"))
        scores = []
        
        for i, img_path in enumerate(tqdm(img_files, desc=cond_name)):
            img = Image.open(img_path).convert("RGB")
            prompt = prompts[i] if i < len(prompts) else prompts[0]
            
            with torch.no_grad():
                inputs = processor(
                    text=[prompt], images=[img],
                    return_tensors="pt", padding=True
                ).to("cuda")
                
                outputs = model(**inputs)
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                score = (img_emb * txt_emb).sum(dim=-1).item()
                scores.append(score)
        
        all_scores[cond_name] = scores
        print(f"  CLIP = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Paired analysis
    baseline_key = "baseline"
    if baseline_key not in all_scores:
        baseline_key = list(all_scores.keys())[0]
    
    baseline_scores = np.array(all_scores[baseline_key])
    
    print("\n" + "="*70)
    print("MULTILAYER INTERVENTION ANALYSIS")
    print("="*70)
    print(f"{'Condition':<25} {'CLIP':>10} {'Δ':>10} {'p-value':>10}")
    print("-"*70)
    
    from scipy import stats
    
    results = {}
    for cond_name, scores in all_scores.items():
        scores = np.array(scores)
        delta = scores - baseline_scores
        mean_delta = np.mean(delta)
        
        if cond_name != baseline_key:
            _, p_value = stats.ttest_1samp(delta, 0)
        else:
            p_value = 1.0
        
        results[cond_name] = {
            "clip_mean": float(np.mean(scores)),
            "delta_mean": float(mean_delta),
            "p_value": float(p_value),
        }
        
        sig = "*" if p_value < 0.05 else ""
        print(f"{cond_name:<25} {np.mean(scores):>10.4f} {mean_delta:>+10.4f} {p_value:>10.4f}{sig}")
    
    print("-"*70)
    
    # 关键比较
    if "single_layer_12" in results and "multi_layer_6_12_18" in results:
        single_delta = results["single_layer_12"]["delta_mean"]
        multi_delta = results["multi_layer_6_12_18"]["delta_mean"]
        
        print(f"\nKey comparison:")
        print(f"  Single layer (12):      Δ = {single_delta:+.4f}")
        print(f"  Multi layer (6+12+18):  Δ = {multi_delta:+.4f}")
        
        if abs(multi_delta) < 0.01 and results["multi_layer_6_12_18"]["p_value"] > 0.05:
            print("\n✓ CONCLUSION: Even multi-layer intervention does not affect quality")
            print("  → Sinks are non-functional across all tested layers")
    
    # 保存
    with open(results_path / "multilayer_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sd3")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results_multilayer")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing results")
    
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate_multilayer(args.output_dir)
    else:
        run_multilayer_experiment(
            model_name=args.model,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            seed=args.seed,
            prompts_file=args.prompts,
            output_dir=args.output_dir,
        )
