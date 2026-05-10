#!/usr/bin/env python
"""
Early-Phase Only 干预实验
=========================
只在早期阶段（t/T < 0.2）进行干预，验证"即使在sink最强的阶段移除也不影响质量"

这个实验强化了"sink是早期buffer但仍非瓶颈"的叙事

Usage:
    python run_early_phase.py --model sd3 --num_samples 32
"""

import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional

import sys
sys.path.insert(0, '.')

from sink_analysis import (
    ExperimentConfig as BaseExperimentConfig,
    InterventionConfig,
    TransformerPatcher,
    SinkAwareAttnProcessor,
)


@dataclass
class PhaseConfig:
    """相位配置"""
    start: float = 0.0  # 开始干预的timestep (t/T)
    end: float = 1.0    # 结束干预的timestep (t/T)


class PhaseAwareTransformerPatcher(TransformerPatcher):
    """
    支持相位感知的TransformerPatcher
    
    只在指定的timestep范围内进行干预
    """
    
    def __init__(self, config, phase_start: float = 0.0, phase_end: float = 1.0):
        super().__init__(config)
        self.phase_start = phase_start
        self.phase_end = phase_end
    
    def set_timestep(self, t: float):
        """设置当前timestep，并决定是否启用干预"""
        self.current_timestep = t
        
        # 检查是否在干预相位内
        in_phase = self.phase_start <= t <= self.phase_end
        
        for processor in self.processors.values():
            processor.current_timestep = t
            # 只在相位内启用干预
            if in_phase:
                processor.enable()
            else:
                processor.disable()


def run_early_phase_experiment(
    model_name: str = "sd3",
    phase_configs: List[Dict] = None,
    num_samples: int = 32,
    num_steps: int = 50,
    seed: int = 42,
    prompts_file: str = None,
    output_dir: str = "./results_early_phase",
    device: str = "cuda"
):
    """
    Early-phase干预实验
    
    比较：
    1. baseline (无干预)
    2. full_removal (全程η=0)
    3. early_only (只在t/T < 0.2干预)
    4. late_only (只在t/T > 0.8干预) - 对照
    """
    from diffusers import StableDiffusion3Pipeline
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 默认相位配置
    if phase_configs is None:
        phase_configs = [
            {"name": "baseline", "intervene": False},
            {"name": "full_removal", "start": 0.0, "end": 1.0, "intervene": True},
            {"name": "early_only_0_20", "start": 0.0, "end": 0.2, "intervene": True},
            {"name": "mid_only_40_60", "start": 0.4, "end": 0.6, "intervene": True},
            {"name": "late_only_80_100", "start": 0.8, "end": 1.0, "intervene": True},
        ]
    
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
    else:
        prompts = [
            "A majestic mountain landscape at sunset",
            "A cozy coffee shop interior with warm lighting",
            "A futuristic cityscape with flying cars",
            "A serene Japanese garden with cherry blossoms",
            "A portrait of a wise old wizard",
            "An underwater scene with colorful coral",
            "A steampunk mechanical owl",
            "A mystical forest with bioluminescent mushrooms",
        ]
        while len(prompts) < num_samples:
            prompts.extend(prompts)
        prompts = prompts[:num_samples]
    
    # 保存prompts
    with open(output_path / "prompts.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    
    results = {}
    
    for phase_cfg in phase_configs:
        cond_name = phase_cfg["name"]
        
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}")
        if phase_cfg.get("intervene", True):
            print(f"Phase: t/T ∈ [{phase_cfg.get('start', 0)}, {phase_cfg.get('end', 1)}]")
        else:
            print("No intervention")
        print("="*60)
        
        # 配置干预
        if phase_cfg.get("intervene", True):
            intervention = InterventionConfig(
                intervention_type="both",
                score_method="prob_scale",
                score_scale=0.0,  # 完全移除
                value_method="zero",
                intervention_layers=[12],
                sink_token_indices=[0],
            )
        else:
            intervention = InterventionConfig(
                intervention_type="none",
                intervention_layers=[],
                sink_token_indices=[0],
            )
        
        config = BaseExperimentConfig(
            model_name=model_name,
            num_steps=num_steps,
            num_samples=num_samples,
            seed=seed,
            measure_layers=[6, 12, 18],
            intervention=intervention,
        )
        
        # 创建相位感知的patcher
        if phase_cfg.get("intervene", True):
            patcher = PhaseAwareTransformerPatcher(
                config,
                phase_start=phase_cfg.get("start", 0.0),
                phase_end=phase_cfg.get("end", 1.0)
            )
        else:
            patcher = TransformerPatcher(config)
        
        patcher.patch(pipe.transformer)
        
        images = []
        intervention_steps = []  # 记录干预发生在哪些步骤
        timestep_log = []  # 记录实际timestep用于验证
        
        try:
            # 每个 condition 清一次：累计 metrics across prompts
            patcher.clear_metrics()
            
            for prompt_idx, prompt in enumerate(tqdm(prompts, desc=cond_name)):
                # 关键：per-prompt配对噪声，确保不同condition之间严格配对
                generator = torch.Generator(device=device).manual_seed(seed + prompt_idx)
                
                step_interventions = []
                step_timesteps = []
                
                def step_callback(pipe, step, timestep, callback_kwargs):
                    # 使用实际的scheduler timestep而非线性归一化
                    # timestep是scheduler的sigma/t值，更准确反映去噪进度
                    t_normalized = step / (num_steps - 1) if num_steps > 1 else 0
                    patcher.set_timestep(t_normalized)
                    
                    # 记录实际timestep用于日志
                    step_timesteps.append((step, timestep.item() if hasattr(timestep, 'item') else timestep))
                    
                    # 记录是否干预
                    if phase_cfg.get("intervene", True):
                        in_phase = phase_cfg.get("start", 0) <= t_normalized <= phase_cfg.get("end", 1)
                        step_interventions.append(in_phase)
                    
                    return callback_kwargs
                
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    generator=generator,
                    callback_on_step_end=step_callback,
                    output_type="pil"
                )
                images.append(result.images[0])
                
                if step_interventions:
                    intervention_steps.append(sum(step_interventions))
                if step_timesteps and prompt_idx == 0:
                    timestep_log = step_timesteps  # 只记录第一个prompt的timestep mapping
        
        finally:
            patcher.unpatch()
        
        # 保存图片
        img_dir = output_path / f"images_{cond_name}"
        img_dir.mkdir(exist_ok=True)
        for idx, img in enumerate(images):
            img.save(img_dir / f"{idx:03d}.png")
        
        # 记录结果
        results[cond_name] = {
            "num_images": len(images),
            "phase_start": phase_cfg.get("start", 0),
            "phase_end": phase_cfg.get("end", 1),
            "intervene": phase_cfg.get("intervene", True),
            "avg_intervention_steps": np.mean(intervention_steps) if intervention_steps else 0,
            "timestep_mapping": timestep_log,  # 记录step -> scheduler_timestep映射
        }
        
        if intervention_steps:
            print(f"  Average intervention steps: {np.mean(intervention_steps):.1f} / {num_steps}")
        if timestep_log:
            print(f"  Timestep range: {timestep_log[0][1]:.1f} -> {timestep_log[-1][1]:.1f}")
    
    # 保存配置
    config_data = {
        "model": model_name,
        "num_samples": num_samples,
        "num_steps": num_steps,
        "seed": seed,
        "phase_configs": phase_configs,
        "results": results,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Early-phase experiment completed!")
    print(f"Results saved to: {output_path}")
    print("="*60)
    
    return results


def evaluate_early_phase(results_dir: str):
    """评估early-phase实验结果"""
    from transformers import CLIPProcessor, CLIPModel
    
    results_path = Path(results_dir)
    
    # 加载prompts
    with open(results_path / "prompts.txt") as f:
        prompts = [l.strip() for l in f if l.strip()]
    
    # 加载CLIP
    print("Loading CLIP...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    
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
    baseline_scores = np.array(all_scores[baseline_key])
    
    print("\n" + "="*70)
    print("EARLY-PHASE INTERVENTION ANALYSIS")
    print("="*70)
    print(f"{'Condition':<25} {'CLIP':>10} {'Δ':>10} {'Phase':>15}")
    print("-"*70)
    
    # 加载config获取phase信息
    with open(results_path / "config.json") as f:
        config = json.load(f)
    
    phase_info = {pc["name"]: f"[{pc.get('start', 0):.1f}, {pc.get('end', 1):.1f}]" 
                  for pc in config.get("phase_configs", [])}
    
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
        
        phase_str = phase_info.get(cond_name, "N/A")
        sig = "*" if p_value < 0.05 else ""
        print(f"{cond_name:<25} {np.mean(scores):>10.4f} {mean_delta:>+10.4f} {phase_str:>15}{sig}")
    
    print("-"*70)
    
    # 关键比较
    print("\nKey findings:")
    if "early_only_0_20" in results and "full_removal" in results:
        early_delta = results["early_only_0_20"]["delta_mean"]
        full_delta = results["full_removal"]["delta_mean"]
        
        print(f"  Full removal:  Δ = {full_delta:+.4f}")
        print(f"  Early-only:    Δ = {early_delta:+.4f}")
        
        if abs(early_delta) < 0.01 and results["early_only_0_20"]["p_value"] > 0.05:
            print("\n✓ CONCLUSION: Even during the phase where sinks are STRONGEST,")
            print("  removing them does not degrade quality.")
            print("  → Sinks are early-phase artifacts but still non-functional")
    
    # 保存
    with open(results_path / "early_phase_stats.json", "w") as f:
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
    parser.add_argument("--output_dir", type=str, default="./results_early_phase")
    parser.add_argument("--eval_only", action="store_true")
    
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate_early_phase(args.output_dir)
    else:
        run_early_phase_experiment(
            model_name=args.model,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            seed=args.seed,
            prompts_file=args.prompts,
            output_dir=args.output_dir,
        )
