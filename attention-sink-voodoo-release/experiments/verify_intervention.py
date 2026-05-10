#!/usr/bin/env python
"""
验证干预效果：检查不同η下sink ratio的实际变化
================================================================
证明η=0确实把sink ratio从~5%压到接近0

这个脚本在每个干预强度下：
1. 生成图片时记录attention metrics
2. 输出 sink_ratio vs η 的关系
"""

import torch
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt

# 复用sink_analysis的核心组件
import sys
sys.path.insert(0, '.')

from sink_analysis import (
    ExperimentConfig, 
    InterventionConfig, 
    TransformerPatcher,
    SinkMetrics
)


def verify_intervention_effect(
    model_name: str = "sd3",
    eta_values: List[float] = [1.0, 0.5, 0.1, 0.01, 0.0],
    num_samples: int = 4,
    num_steps: int = 20,
    output_dir: str = "./verify_intervention",
    device: str = "cuda"
):
    """
    验证不同η值下sink ratio的实际变化
    """
    from diffusers import StableDiffusion3Pipeline
    import torch
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"Loading {model_name} pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    ).to(device)
    
    test_prompts = [
        "A beautiful mountain landscape at sunset",
        "A cute cat sitting on a windowsill",
        "A futuristic city with flying cars",
        "A cozy coffee shop interior",
    ][:num_samples]
    
    results = {}
    
    for eta in eta_values:
        print(f"\n{'='*60}")
        print(f"Testing η = {eta}")
        print("="*60)
        
        # 配置
        intervention = InterventionConfig(
            intervention_type="score_only" if eta < 1.0 else "none",
            score_method="prob_scale",
            score_scale=eta,
            intervention_layers=[12],  # 中间层
            sink_token_indices=[0],
        )
        
        config = ExperimentConfig(
            model_name=model_name,
            num_steps=num_steps,
            num_samples=num_samples,
            seed=42,
            measure_layers=[12],  # 只测量干预层
            intervention=intervention,
        )
        
        # Patch
        patcher = TransformerPatcher(config)
        patcher.patch(pipe.transformer)
        
        all_sink_ratios = []
        
        try:
            generator = torch.Generator(device=device).manual_seed(42)
            
            for i, prompt in enumerate(test_prompts):
                print(f"  Sample {i+1}/{len(test_prompts)}: {prompt[:50]}...")
                
                patcher.clear_metrics()
                
                def step_callback(pipe, step, timestep, callback_kwargs):
                    patcher.set_timestep(step / num_steps)
                    return callback_kwargs
                
                _ = pipe(
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    generator=generator,
                    callback_on_step_end=step_callback,
                    output_type="pil"
                )
                
                # 收集metrics
                df = patcher.get_metrics_dataframe()
                if len(df) > 0:
                    mean_sink = df['sink_ratio'].mean()
                    all_sink_ratios.append(mean_sink)
                    print(f"    Sink ratio (mean): {mean_sink:.4f}")
        
        finally:
            patcher.unpatch()
        
        # 汇总
        if all_sink_ratios:
            mean_ratio = np.mean(all_sink_ratios)
            std_ratio = np.std(all_sink_ratios)
            results[eta] = {
                "mean_sink_ratio": float(mean_ratio),
                "std_sink_ratio": float(std_ratio),
                "per_sample": all_sink_ratios,
            }
            print(f"\n  η={eta}: Sink ratio = {mean_ratio:.4f} ± {std_ratio:.4f}")
    
    # 计算reduction factor
    if 1.0 in results and 0.0 in results:
        baseline_ratio = results[1.0]["mean_sink_ratio"]
        zero_ratio = results[0.0]["mean_sink_ratio"]
        reduction = baseline_ratio / max(zero_ratio, 1e-6)
        print(f"\n{'='*60}")
        print(f"REDUCTION FACTOR: {reduction:.1f}x")
        print(f"  Baseline (η=1.0): {baseline_ratio:.4f}")
        print(f"  Zero (η=0.0): {zero_ratio:.6f}")
        print("="*60)
        results["reduction_factor"] = float(reduction)
    
    # 保存结果
    json_path = output_path / "intervention_verification.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")
    
    # 绘图
    plot_intervention_verification(results, output_path / "intervention_verification.png")
    
    return results


def plot_intervention_verification(results: dict, save_path: str):
    """绘制η vs sink_ratio曲线"""
    
    # 提取数据
    etas = []
    ratios = []
    stds = []
    
    for eta, data in results.items():
        if isinstance(eta, (int, float)):
            etas.append(eta)
            ratios.append(data["mean_sink_ratio"])
            stds.append(data["std_sink_ratio"])
    
    # 排序
    sorted_data = sorted(zip(etas, ratios, stds))
    etas, ratios, stds = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(etas, ratios, yerr=stds, fmt='o-', capsize=5, 
                linewidth=2, markersize=10, color='#E91E63')
    
    ax.set_xlabel('Score Scale (η)', fontsize=12)
    ax.set_ylabel('Sink Attention Ratio', fontsize=12)
    ax.set_title('Intervention Verification: η vs Sink Ratio', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 标注reduction
    if len(etas) >= 2:
        reduction = ratios[-1] / max(ratios[0], 1e-6)  # η=1 / η=0
        ax.text(0.5, max(ratios) * 0.8, 
               f"Reduction: {reduction:.1f}x\n(η=1.0 → η=0.0)",
               fontsize=11, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved verification plot to {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./verify_intervention")
    args = parser.parse_args()
    
    verify_intervention_effect(
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
    )
