"""
Attention Sink Experiments (v2)
===============================
统一的实验入口，支持：
- H1: Sink演化曲线
- H2: Score vs Value因果干预对比
- Sweep: 干预强度扫描

Usage:
    # H1实验
    python run_experiment.py --model sd3 --experiment h1 --steps 50 --num_samples 16
    
    # H2实验（推荐配置）
    python run_experiment.py --model sd3 --experiment h2 --steps 50 --num_samples 16
    
    # 干预强度sweep
    python run_experiment.py --model sd3 --experiment sweep --sweep_type score
    
    # 快速测试
    python run_experiment.py --model sd3 --experiment h1 --quick
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict
from tqdm import tqdm

from sink_analysis import (
    ExperimentConfig,
    InterventionConfig,
    TransformerPatcher,
    SinkAwareAttnProcessor,
    plot_h1_curves,
    plot_intervention_sweep,
)


# =============================================================================
# 1. Model Loading
# =============================================================================

def load_pipeline(model_name: str, device: str = "cuda", dtype=None):
    """
    加载diffusion pipeline
    """
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading {model_name} pipeline...")
    
    if model_name == "sd3":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=dtype,
        ).to(device)
        
    elif model_name == "sd3.5":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=dtype,
        ).to(device)
        
    elif model_name == "flux-schnell":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        ).to(device)
        
    elif model_name == "flux-dev":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        ).to(device)
        
    elif model_name == "dit":
        # DiT需要特殊处理，这里用placeholder
        raise NotImplementedError("DiT requires custom loading - implement based on your setup")
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 获取层数
    num_layers = len(pipe.transformer.transformer_blocks)
    print(f"Model loaded: {num_layers} transformer blocks")
    
    return pipe, num_layers


def get_default_prompts(n: int = 32) -> List[str]:
    """默认prompt列表"""
    base_prompts = [
        "A majestic mountain landscape at sunset with snow-capped peaks",
        "A cozy coffee shop interior with warm lighting and wooden furniture", 
        "A futuristic cityscape with flying cars and neon lights",
        "A serene Japanese garden with cherry blossoms and a koi pond",
        "A portrait of a wise old wizard with a long white beard",
        "An underwater scene with colorful coral reefs and tropical fish",
        "A steampunk mechanical owl with brass gears and glowing eyes",
        "A mystical forest with bioluminescent mushrooms and fireflies",
        "A vintage car parked on a rainy city street at night",
        "An astronaut floating in space with Earth in the background",
        "A medieval castle on a cliff overlooking a stormy sea",
        "A cozy library with floor-to-ceiling bookshelves and a fireplace",
        "A dragon perched on a mountain peak at dawn",
        "A bustling marketplace in ancient Morocco",
        "A hyperrealistic portrait of a cat wearing a crown",
        "A cyberpunk street scene with holographic advertisements",
    ]
    
    # 循环扩展到需要的数量
    prompts = []
    while len(prompts) < n:
        prompts.extend(base_prompts)
    return prompts[:n]


def load_prompts_from_file(filepath: str, n: int = None) -> List[str]:
    """从文件读取prompts

    Args:
        filepath: prompts文件路径，每行一个prompt
        n: 最多返回n个prompts，None表示全部

    Returns:
        prompts列表
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    if n is not None and n < len(prompts):
        prompts = prompts[:n]

    print(f"Loaded {len(prompts)} prompts from {filepath}")
    return prompts


# =============================================================================
# 2. Experiment Runners
# =============================================================================

def run_h1_experiment(
    pipe,
    config: ExperimentConfig,
    prompts: List[str],
    output_dir: Path,
    device: str = "cuda"
) -> Dict:
    """
    H1实验：测量sink strength随timestep的演化

    对于Flux模型：由于无法安全patch，只生成图片作为baseline
    """
    print("\n" + "="*60)
    print("H1 Experiment: Sink Evolution Curves")
    print("="*60)

    # 创建patcher（使用AttnProcessor API）
    patcher = TransformerPatcher(config)
    patcher.patch(pipe.transformer)

    try:
        images = []

        for i, prompt in enumerate(tqdm(prompts[:config.num_samples], desc="Generating")):
            # Per-sample seed
            sample_seed = config.seed + i
            generator = torch.Generator(device=device).manual_seed(sample_seed)

            # Step callback
            def step_callback(pipe, step, timestep, callback_kwargs):
                t_normalized = step / config.num_steps
                patcher.set_timestep(t_normalized)
                return callback_kwargs

            result = pipe(
                prompt=prompt,
                num_inference_steps=config.num_steps,
                generator=generator,
                callback_on_step_end=step_callback,
                output_type="pil"
            )
            images.append(result.images[0])

        # 保存图片
        img_dir = output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for idx, img in enumerate(images):
            img.save(img_dir / f"{idx:03d}.png")
        print(f"Saved {len(images)} images to {img_dir}")

        # 保存结果
        df = patcher.get_metrics_dataframe()

        if len(df) > 0:
            df.to_csv(output_dir / "h1_metrics.csv", index=False)

            # 绘图
            plot_h1_curves(df, str(output_dir / "h1_curves.png"))

            # 统计摘要
            summary = df.groupby('layer')[['sink_ratio', 'entropy', 'max_act']].agg(['mean', 'std'])
            print("\nH1 Summary:")
            print(summary)

            return {"dataframe": df, "summary": summary, "images": images}
        else:
            print("\nNo metrics collected (Flux mode - patching skipped)")
            print("Images generated as baseline for quality comparison")
            return {"dataframe": df, "images": images}

    finally:
        patcher.unpatch()


def run_h2_experiment(
    pipe,
    config: ExperimentConfig,
    prompts: List[str],
    output_dir: Path,
    device: str = "cuda",
    score_scale: float = 0.1,
    value_method: str = "zero",
) -> Dict:
    """
    H2实验：对比 baseline / score-only / value-only 三种条件

    使用per-sample seed对齐确保严格配对设计
    """
    print("\n" + "="*60)
    print("H2 Experiment: Score vs Value Causal Intervention")
    print("="*60)
    print(f"Score scale: {score_scale}")
    print(f"Value method: {value_method}")

    results = {}

    # 条件定义：(name, intervention_type, score_scale, value_method)
    # 注意：none条件使用中性值（score_scale=1.0表示不缩放）
    conditions = [
        ("none", "none", 1.0, "zero"),  # 中性值：scale=1.0不改变任何东西
        ("score_only", "score_only", score_scale, "zero"),
        ("value_only", "value_only", 1.0, value_method),
    ]

    num_samples = min(config.num_samples, len(prompts))

    for cond_name, intervention_type, s_scale, v_method in conditions:
        print(f"\n--- Condition: {cond_name} ---")

        # 重置attention调用计数器
        SinkAwareAttnProcessor.reset_counters()

        # 创建配置
        intervention = InterventionConfig(
            intervention_type=intervention_type,
            score_method="prob_scale",
            score_scale=s_scale,
            value_method=v_method,
            intervention_layers=config.intervention.intervention_layers,
            sink_token_indices=config.intervention.sink_token_indices,
        )

        cond_config = ExperimentConfig(
            model_name=config.model_name,
            num_steps=config.num_steps,
            num_samples=num_samples,
            seed=config.seed,
            measure_layers=config.measure_layers,
            intervention=intervention,
        )

        # Patch
        patcher = TransformerPatcher(cond_config)
        patcher.patch(pipe.transformer)

        try:
            images = []

            for i, prompt in enumerate(tqdm(prompts[:num_samples], desc=cond_name)):
                # Per-sample seed对齐：每个样本使用 (base_seed + sample_index)
                # 这确保了跨条件的严格配对
                sample_seed = config.seed + i
                generator = torch.Generator(device=device).manual_seed(sample_seed)

                def step_callback(pipe, step, timestep, callback_kwargs):
                    patcher.set_timestep(step / config.num_steps)
                    return callback_kwargs

                result = pipe(
                    prompt=prompt,
                    num_inference_steps=config.num_steps,
                    generator=generator,
                    callback_on_step_end=step_callback,
                    output_type="pil"
                )
                images.append(result.images[0])

            # 保存图片
            img_dir = output_dir / f"images_{cond_name}"
            img_dir.mkdir(exist_ok=True)
            for idx, img in enumerate(images):
                img.save(img_dir / f"{idx:03d}.png")

            # 保存metrics
            metrics_df = patcher.get_metrics_dataframe()
            metrics_df.to_csv(output_dir / f"metrics_{cond_name}.csv", index=False)

            results[cond_name] = {
                "images": images,
                "metrics_df": metrics_df,
                "config": asdict(cond_config),
            }

            print(f"  Saved {len(images)} images to {img_dir}")

            # 打印attention调用统计
            SinkAwareAttnProcessor.print_stats()

        finally:
            patcher.unpatch()

    # 保存prompts供后续评估
    with open(output_dir / "prompts.txt", "w") as f:
        for p in prompts[:num_samples]:
            f.write(p + "\n")

    return results


def run_sweep_experiment(
    pipe,
    config: ExperimentConfig,
    prompts: List[str],
    output_dir: Path,
    device: str = "cuda",
    sweep_type: str = "score",  # "score" or "value"
    sweep_values: List[float] = None,
) -> Dict:
    """
    干预强度sweep实验
    """
    print("\n" + "="*60)
    print(f"Sweep Experiment: {sweep_type} intervention strength")
    print("="*60)

    if sweep_values is None:
        if sweep_type == "score":
            # η: 0 = 完全移除sink, 1 = 不干预
            # 包含激进值0.01用于剂量曲线
            sweep_values = [0.0, 0.01, 0.1, 0.25, 0.5, 1.0]
        else:
            # Value sweep: 不同方法
            # 使用特殊编码: -1=zero, -2=mean, 0-1=lerp_alpha
            sweep_values = [-1, -2, 0.0, 0.5, 1.0]  # zero, mean, lerp_0, lerp_0.5, no_intervention

    results = {}

    for val in sweep_values:
        # 配置干预
        if sweep_type == "score":
            intervention = InterventionConfig(
                intervention_type="score_only",
                score_method="prob_scale",
                score_scale=val,
                intervention_layers=config.intervention.intervention_layers,
                sink_token_indices=config.intervention.sink_token_indices,
            )
            val_label = f"η={val}"
        else:  # value
            # 特殊编码: -1=zero, -2=mean, 0-1=lerp_alpha
            if val == -1:
                method, alpha, val_label = "zero", 0.0, "zero"
            elif val == -2:
                method, alpha, val_label = "mean", 0.0, "mean"
            elif val == 1.0:
                # α=1.0 表示不干预，使用none
                intervention = InterventionConfig(
                    intervention_type="none",
                    intervention_layers=config.intervention.intervention_layers,
                    sink_token_indices=config.intervention.sink_token_indices,
                )
                val_label = "baseline"
            else:
                method, alpha, val_label = "lerp", val, f"lerp_{val}"

            if val != 1.0:
                intervention = InterventionConfig(
                    intervention_type="value_only",
                    value_method=method,
                    value_lerp_alpha=alpha,
                    intervention_layers=config.intervention.intervention_layers,
                    sink_token_indices=config.intervention.sink_token_indices,
                )

        print(f"\n--- {sweep_type}: {val_label} ---")

        sweep_config = ExperimentConfig(
            model_name=config.model_name,
            num_steps=config.num_steps,
            num_samples=config.num_samples,
            seed=config.seed,
            measure_layers=config.measure_layers,
            intervention=intervention,
        )

        patcher = TransformerPatcher(sweep_config)
        patcher.patch(pipe.transformer)

        try:
            generator = torch.Generator(device=device).manual_seed(config.seed)
            images = []

            for prompt in tqdm(prompts[:config.num_samples], desc=val_label):
                def step_callback(pipe, step, timestep, callback_kwargs):
                    patcher.set_timestep(step / config.num_steps)
                    return callback_kwargs

                result = pipe(
                    prompt=prompt,
                    num_inference_steps=config.num_steps,
                    generator=generator,
                    callback_on_step_end=step_callback,
                    output_type="pil"
                )
                images.append(result.images[0])

            # 保存 - 使用val_label作为目录名
            safe_label = val_label.replace("=", "_").replace(".", "p")
            img_dir = output_dir / f"images_{safe_label}"
            img_dir.mkdir(exist_ok=True)
            for idx, img in enumerate(images):
                img.save(img_dir / f"{idx:03d}.png")

            results[val_label] = {
                "images": images,
                "metrics_df": patcher.get_metrics_dataframe(),
                "val": val,
            }

        finally:
            patcher.unpatch()

    # 保存prompts
    with open(output_dir / "prompts.txt", "w") as f:
        for p in prompts[:config.num_samples]:
            f.write(p + "\n")

    return results


# =============================================================================
# 3. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Attention Sink Experiments v2")

    # Model
    parser.add_argument("--model", type=str, default="sd3",
                       choices=["sd3", "sd3.5", "flux-schnell", "flux-dev", "dit"])
    parser.add_argument("--device", type=str, default="cuda")

    # Experiment type
    parser.add_argument("--experiment", type=str, default="h1",
                       choices=["h1", "h2", "sweep"])

    # Common params
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--prompts", type=str, default=None,
                       help="Path to prompts file (one prompt per line). If not provided, uses default prompts.")

    # H2 specific
    parser.add_argument("--score_scale", type=float, default=0.1,
                       help="Score intervention scale (0=remove, 1=keep)")
    parser.add_argument("--value_method", type=str, default="zero",
                       choices=["zero", "mean", "noise", "lerp"])

    # Sweep specific
    parser.add_argument("--sweep_type", type=str, default="score",
                       choices=["score", "value"])

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.steps = 20
        args.num_samples = 4

    # Output dir
    output_dir = Path(args.output_dir) / f"{args.experiment}_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Attention Sink Experiment v2")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Experiment: {args.experiment}")
    print(f"Steps: {args.steps}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {output_dir}")
    print("="*60)

    # Load model
    pipe, num_layers = load_pipeline(args.model, args.device)

    # 确定测量和干预层
    measure_layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
    intervention_layers = [num_layers // 2]  # 中间层

    print(f"Measure layers: {measure_layers}")
    print(f"Intervention layers: {intervention_layers}")

    # 创建配置
    intervention_config = InterventionConfig(
        intervention_layers=intervention_layers,
        sink_token_indices=[0],
    )

    config = ExperimentConfig(
        model_name=args.model,
        num_steps=args.steps,
        num_samples=args.num_samples,
        seed=args.seed,
        measure_layers=measure_layers,
        intervention=intervention_config,
    )

    # 保存配置
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Prompts - 从文件加载或使用默认
    if args.prompts:
        prompts = load_prompts_from_file(args.prompts, args.num_samples)
    else:
        prompts = get_default_prompts(args.num_samples)
        print(f"Using {len(prompts)} default prompts (循环使用16个基础prompts)")
    
    # Run experiment
    if args.experiment == "h1":
        results = run_h1_experiment(pipe, config, prompts, output_dir, args.device)
        
    elif args.experiment == "h2":
        results = run_h2_experiment(
            pipe, config, prompts, output_dir, args.device,
            score_scale=args.score_scale,
            value_method=args.value_method,
        )
        print("\n" + "="*60)
        print("Next: Run quality evaluation")
        print(f"  python quality_metrics.py --results_dir {output_dir}")
        print("="*60)
        
    elif args.experiment == "sweep":
        results = run_sweep_experiment(
            pipe, config, prompts, output_dir, args.device,
            sweep_type=args.sweep_type,
        )
        print("\n" + "="*60)
        print("Next: Run quality evaluation for sweep")
        print(f"  python quality_metrics.py --results_dir {output_dir} --sweep")
        print("="*60)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()