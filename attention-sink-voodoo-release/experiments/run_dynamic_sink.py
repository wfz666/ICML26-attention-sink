#!/usr/bin/env python
"""
Dynamic Sink Intervention for SD3 (JointAttnProcessor2_0 Compatible)
=====================================================================
正确理解SD3的joint attention结构:

Image attention path:
  query_img = to_q(hidden_states)                    # [B, 4096, D]
  key = concat([to_k(img), add_k_proj(txt)])         # [B, 4429, D]  
  value = concat([to_v(img), add_v_proj(txt)])       # [B, 4429, D]
  attn_img = softmax(query_img @ key.T) @ value      # [B, 4096, D]

Text attention path (shares K, V):
  query_txt = add_q_proj(encoder_hidden_states)      # [B, 333, D]
  attn_txt = softmax(query_txt @ key.T) @ value      # [B, 333, D]

Return: (attn_img, attn_txt)

所以Query是分开的，K/V是共享的！

Usage:
    python run_dynamic_sink.py --num_samples 32 --top_k 1 --layers 12
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import Attention


@dataclass
class DynamicSinkConfig:
    """Dynamic sink实验配置"""
    model_name: str = "sd3"
    num_steps: int = 20
    num_samples: int = 32
    seed: int = 42
    
    top_k: int = 1
    intervention_layers: List[int] = field(default_factory=lambda: [12])
    measure_layers: List[int] = field(default_factory=lambda: [6, 12, 18])
    
    mask_value: float = 1e4


class DynamicSinkJointAttnProcessor:
    """
    SD3 JointAttnProcessor2_0 的动态sink干预版本
    
    关键结构:
    - Query分开: img_query [B, 4096, D], txt_query [B, 333, D]
    - K/V共享: key/value [B, 4429, D]
    - 两个独立的attention计算
    - 动态sink在key维度(4429)上检测和干预
    """
    
    def __init__(
        self,
        layer_idx: int,
        top_k: int = 1,
        intervention_enabled: bool = True,
        measure_only: bool = False,
        mask_value: float = 1e4,
    ):
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.intervention_enabled = intervention_enabled
        self.measure_only = measure_only
        self.mask_value = mask_value
        
        self.metrics = defaultdict(list)
        self.current_timestep = 0.0
    
    def set_timestep(self, t: float):
        self.current_timestep = t
    
    def clear_metrics(self):
        self.metrics = defaultdict(list)
    
    def get_metrics_summary(self) -> Dict:
        summary = {}
        for key, values in self.metrics.items():
            if values:
                arr = np.array(values)
                summary[key] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                }
        return summary
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SD3 Joint Attention with dynamic sink intervention.
        """
        
        batch_size = hidden_states.shape[0]
        
        # ===== Image Query =====
        query_img = attn.to_q(hidden_states)  # [B, 4096, D]
        
        # ===== Image Key/Value =====
        key_img = attn.to_k(hidden_states)    # [B, 4096, D]
        value_img = attn.to_v(hidden_states)  # [B, 4096, D]
        
        inner_dim = key_img.shape[-1]
        head_dim = inner_dim // attn.heads
        
        # ===== Text Query/Key/Value (SD3 specific) =====
        if encoder_hidden_states is not None:
            query_txt = attn.add_q_proj(encoder_hidden_states)  # [B, 333, D]
            key_txt = attn.add_k_proj(encoder_hidden_states)    # [B, 333, D]
            value_txt = attn.add_v_proj(encoder_hidden_states)  # [B, 333, D]
            
            # Concatenate K and V (shared between both attention paths)
            key = torch.cat([key_img, key_txt], dim=1)      # [B, 4429, D]
            value = torch.cat([value_img, value_txt], dim=1)  # [B, 4429, D]
        else:
            query_txt = None
            key = key_img
            value = value_img
        
        # ===== Reshape for multi-head attention =====
        # [B, N, D] -> [B, H, N, D/H]
        query_img = query_img.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        if query_txt is not None:
            query_txt = query_txt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # ===== Normalization (SD3 uses QK normalization) =====
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            query_img = attn.norm_q(query_img)
        if hasattr(attn, 'norm_k') and attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # Normalize text query if exists
        if query_txt is not None and hasattr(attn, 'norm_added_q') and attn.norm_added_q is not None:
            query_txt = attn.norm_added_q(query_txt)
        
        # ===== Compute image attention logits =====
        scale = head_dim ** -0.5
        attn_logits_img = torch.matmul(query_img, key.transpose(-2, -1)) * scale
        # attn_logits_img: [B, H, 4096, 4429]
        
        # ===== Compute baseline attention weights (for dynamic sink detection) =====
        attn_logits_img_f32 = attn_logits_img.float()
        attn_weights_img_baseline = F.softmax(attn_logits_img_f32, dim=-1)
        
        # ===== Compute incoming mass for dynamic sink detection =====
        # incoming_mass[j] = mean over queries of A[:, :, i, j]
        # This tells us which KEY positions receive the most attention
        incoming_mass = attn_weights_img_baseline.mean(dim=2)  # [B, H, 4429]
        
        # ===== Find dynamic top-k sinks (in KEY dimension) =====
        topk_values, topk_indices = torch.topk(incoming_mass, k=self.top_k, dim=-1)
        
        # ===== Record metrics =====
        with torch.no_grad():
            top1_mass = incoming_mass.max(dim=-1).values.mean().item()
            self.metrics['top1_incoming_mass_before'].append(top1_mass)
            
            topk_total = topk_values.sum(dim=-1).mean().item()
            self.metrics[f'top{self.top_k}_total_mass_before'].append(topk_total)
            
            # Check if index-0 is among dynamic sinks
            is_index0_sink = (topk_indices == 0).any(dim=-1).float().mean().item()
            self.metrics['index0_is_dynamic_sink_ratio'].append(is_index0_sink)
            
            entropy = -(attn_weights_img_baseline * (attn_weights_img_baseline + 1e-10).log()).sum(dim=-1).mean().item()
            self.metrics['entropy_before'].append(entropy)
        
        # ===== Apply dynamic sink intervention =====
        if self.intervention_enabled and not self.measure_only:
            # Create mask for top-k KEY positions
            mask = torch.zeros_like(incoming_mass)  # [B, H, 4429]
            mask.scatter_(-1, topk_indices, 1.0)
            mask = mask.unsqueeze(2)  # [B, H, 1, 4429] - broadcast over query dim
            
            # Apply mask to image attention logits
            attn_logits_img_f32 = attn_logits_img_f32 - self.mask_value * mask
            attn_weights_img = F.softmax(attn_logits_img_f32, dim=-1)
            
            # Record post-intervention metrics
            with torch.no_grad():
                incoming_mass_after = attn_weights_img.mean(dim=2)
                masked_mass_after = torch.gather(incoming_mass_after, -1, topk_indices)
                top1_mass_after = masked_mass_after[:, :, 0].mean().item()
                self.metrics['top1_incoming_mass_after'].append(top1_mass_after)
                
                reduction = top1_mass / max(top1_mass_after, 1e-10)
                self.metrics['reduction_factor'].append(reduction)
        else:
            attn_weights_img = attn_weights_img_baseline
            mask = None
        
        # Convert back to original dtype
        attn_weights_img = attn_weights_img.to(value.dtype)
        
        # ===== Compute image attention output =====
        hidden_states = torch.matmul(attn_weights_img, value)
        # hidden_states: [B, H, 4096, D/H]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # hidden_states: [B, 4096, D]
        
        # ===== Compute text attention (if encoder_hidden_states exists) =====
        if query_txt is not None:
            attn_logits_txt = torch.matmul(query_txt, key.transpose(-2, -1)) * scale
            # attn_logits_txt: [B, H, 333, 4429]
            
            attn_logits_txt_f32 = attn_logits_txt.float()
            
            # Apply same mask to text attention (same KEY positions)
            if mask is not None:
                attn_logits_txt_f32 = attn_logits_txt_f32 - self.mask_value * mask
            
            attn_weights_txt = F.softmax(attn_logits_txt_f32, dim=-1)
            attn_weights_txt = attn_weights_txt.to(value.dtype)
            
            encoder_hidden_states_out = torch.matmul(attn_weights_txt, value)
            # encoder_hidden_states_out: [B, H, 333, D/H]
            encoder_hidden_states_out = encoder_hidden_states_out.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            # encoder_hidden_states_out: [B, 333, D]
        else:
            encoder_hidden_states_out = None
        
        # ===== Output projections =====
        # Image output
        if hasattr(attn, 'to_out') and attn.to_out is not None:
            if isinstance(attn.to_out, torch.nn.ModuleList):
                for module in attn.to_out:
                    hidden_states = module(hidden_states)
            else:
                hidden_states = attn.to_out(hidden_states)
        
        # Text output (SD3 specific)
        if encoder_hidden_states_out is not None:
            if hasattr(attn, 'to_add_out') and attn.to_add_out is not None:
                encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)
            return hidden_states, encoder_hidden_states_out
        
        return hidden_states, encoder_hidden_states


class DynamicSinkPatcher:
    """管理Dynamic Sink Processors"""
    
    def __init__(self, config: DynamicSinkConfig):
        self.config = config
        self.processors: Dict[int, DynamicSinkJointAttnProcessor] = {}
        self.original_processors: Dict[int, object] = {}
        self._transformer = None
    
    def patch(self, transformer):
        self._transformer = transformer
        self.original_processors = {}
        self.processors = {}
        
        if not hasattr(transformer, 'transformer_blocks'):
            raise RuntimeError("Transformer does not have 'transformer_blocks'")
        
        num_blocks = len(transformer.transformer_blocks)
        print(f"Found {num_blocks} transformer blocks")
        
        all_layers = set(self.config.intervention_layers) | set(self.config.measure_layers)
        for layer_idx in all_layers:
            if layer_idx >= num_blocks:
                raise RuntimeError(f"Layer {layer_idx} >= {num_blocks}")
        
        for idx, block in enumerate(transformer.transformer_blocks):
            is_intervention = idx in self.config.intervention_layers
            is_measure = idx in self.config.measure_layers
            
            if is_intervention or is_measure:
                if not hasattr(block, 'attn'):
                    continue
                
                self.original_processors[idx] = block.attn.processor
                
                processor = DynamicSinkJointAttnProcessor(
                    layer_idx=idx,
                    top_k=self.config.top_k,
                    intervention_enabled=is_intervention,
                    measure_only=not is_intervention,
                    mask_value=self.config.mask_value,
                )
                
                block.attn.processor = processor
                self.processors[idx] = processor
                
                status = "INTERVENTION" if is_intervention else "measure"
                print(f"  Layer {idx}: {status}")
        
        return self
    
    def unpatch(self):
        if self._transformer is None:
            return
        for idx, proc in self.original_processors.items():
            self._transformer.transformer_blocks[idx].attn.processor = proc
        self.processors = {}
        self.original_processors = {}
    
    def set_timestep(self, t: float):
        for p in self.processors.values():
            p.set_timestep(t)
    
    def clear_metrics(self):
        for p in self.processors.values():
            p.clear_metrics()
    
    def get_all_metrics(self) -> Dict:
        return {f"layer_{idx}": p.get_metrics_summary() for idx, p in self.processors.items()}


def load_prompts(prompts_file: str = None, num_samples: int = 32) -> List[str]:
    default_prompts = [
        "A photo of a cat sitting on a windowsill",
        "A beautiful sunset over the ocean",
        "A professional portrait of a businessman",
        "An abstract painting with vibrant colors",
        "A cozy living room with modern furniture",
        "A mountain landscape with snow-capped peaks",
        "A delicious plate of pasta with tomato sauce",
        "A futuristic city skyline at night",
        "A golden retriever playing in a park",
        "A steaming cup of coffee on a wooden table",
        "A serene Japanese garden with cherry blossoms",
        "An astronaut floating in space",
        "A vintage car on a desert highway",
        "A colorful coral reef underwater",
        "A cozy cabin in snowy mountains",
        "A bustling street market in Asia",
    ]
    
    if prompts_file and Path(prompts_file).exists():
        with open(prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {prompts_file}")
        return prompts[:num_samples]
    
    result = []
    while len(result) < num_samples:
        result.extend(default_prompts)
    return result[:num_samples]


def run_experiment(
    config: DynamicSinkConfig,
    output_dir: Path,
    prompts: List[str],
    device: str = "cuda",
):
    print("="*60)
    print(f"Dynamic Sink Experiment: top-{config.top_k}")
    print(f"Intervention layers: {config.intervention_layers}")
    print(f"Steps: {config.num_steps}, Samples: {len(prompts)}")
    print("="*60)
    
    # Save prompts
    with open(output_dir / "prompts.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    
    print("\nLoading SD3...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to(device)
    
    results = {}
    
    # ===== Baseline =====
    print("\n" + "="*60)
    print("Condition: baseline (measure only)")
    print("="*60)
    
    baseline_cfg = DynamicSinkConfig(
        num_steps=config.num_steps, seed=config.seed, top_k=config.top_k,
        intervention_layers=[], measure_layers=config.measure_layers,
    )
    patcher = DynamicSinkPatcher(baseline_cfg)
    patcher.patch(pipe.transformer)
    patcher.clear_metrics()
    
    baseline_images = []
    for i, prompt in enumerate(tqdm(prompts, desc="baseline")):
        gen = torch.Generator(device=device).manual_seed(config.seed + i)
        def cb(pipe, step, ts, kw):
            patcher.set_timestep(step / max(config.num_steps-1, 1))
            return kw
        img = pipe(prompt, num_inference_steps=config.num_steps, generator=gen,
                   callback_on_step_end=cb, output_type="pil").images[0]
        baseline_images.append(img)
    
    baseline_metrics = patcher.get_all_metrics()
    patcher.unpatch()
    
    img_dir = output_dir / "images_baseline"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(baseline_images):
        img.save(img_dir / f"{i:03d}.png")
    
    results['baseline'] = {'num_images': len(baseline_images), 'metrics': baseline_metrics}
    
    # ===== Intervention =====
    print("\n" + "="*60)
    print(f"Condition: dynamic_top{config.top_k} (intervention)")
    print("="*60)
    
    patcher = DynamicSinkPatcher(config)
    patcher.patch(pipe.transformer)
    patcher.clear_metrics()
    
    intervention_images = []
    for i, prompt in enumerate(tqdm(prompts, desc=f"top{config.top_k}")):
        gen = torch.Generator(device=device).manual_seed(config.seed + i)
        def cb(pipe, step, ts, kw):
            patcher.set_timestep(step / max(config.num_steps-1, 1))
            return kw
        img = pipe(prompt, num_inference_steps=config.num_steps, generator=gen,
                   callback_on_step_end=cb, output_type="pil").images[0]
        intervention_images.append(img)
    
    intervention_metrics = patcher.get_all_metrics()
    patcher.unpatch()
    
    img_dir = output_dir / f"images_dynamic_top{config.top_k}"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(intervention_images):
        img.save(img_dir / f"{i:03d}.png")
    
    cond_name = f"dynamic_top{config.top_k}_layers_{'_'.join(map(str, config.intervention_layers))}"
    results[cond_name] = {
        'num_images': len(intervention_images),
        'metrics': intervention_metrics,
        'intervention_layers': config.intervention_layers,
        'top_k': config.top_k,
    }
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'num_steps': config.num_steps, 'num_samples': len(prompts),
            'seed': config.seed, 'top_k': config.top_k,
            'intervention_layers': config.intervention_layers,
            'prompts': prompts, 'results': results,
        }, f, indent=2)
    
    # Print verification
    print("\n" + "="*60)
    print("INTERVENTION VERIFICATION")
    print("="*60)
    for lk, lm in results[cond_name]['metrics'].items():
        print(f"\n{lk}:")
        if 'top1_incoming_mass_before' in lm:
            print(f"  Before: {lm['top1_incoming_mass_before']['mean']:.4f} ({lm['top1_incoming_mass_before']['mean']*100:.2f}%)")
        if 'top1_incoming_mass_after' in lm:
            print(f"  After:  {lm['top1_incoming_mass_after']['mean']:.6f} ({lm['top1_incoming_mass_after']['mean']*100:.4f}%)")
        if 'reduction_factor' in lm:
            print(f"  Reduction: {lm['reduction_factor']['mean']:,.0f}×")
        if 'index0_is_dynamic_sink_ratio' in lm:
            print(f"  Index-0 is sink: {lm['index0_is_dynamic_sink_ratio']['mean']:.1%}")
    
    return results, baseline_images, intervention_images


def evaluate_clip(images, prompts, device="cuda"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    scores = []
    with torch.no_grad():
        for img, prompt in tqdm(list(zip(images, prompts)), desc="CLIP"):
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_t = preprocess(img).unsqueeze(0).to(device)
            txt_t = tokenizer([prompt]).to(device)
            img_f = model.encode_image(img_t)
            txt_f = model.encode_text(txt_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            scores.append((img_f @ txt_f.T).item())
    return scores


def compute_paired_stats(baseline, intervention):
    from scipy import stats
    b, i = np.array(baseline), np.array(intervention)
    d = i - b
    
    boot_means = [np.mean(d[np.random.choice(len(d), len(d), replace=True)]) for _ in range(1000)]
    _, p = stats.ttest_rel(i, b)
    
    return {
        'baseline_mean': float(np.mean(b)), 
        'baseline_std': float(np.std(b)),
        'intervention_mean': float(np.mean(i)),
        'intervention_std': float(np.std(i)),
        'delta_mean': float(np.mean(d)), 
        'delta_std': float(np.std(d)),
        'ci_lower': float(np.percentile(boot_means, 2.5)),
        'ci_upper': float(np.percentile(boot_means, 97.5)),
        'p_value': float(p),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--layers', type=str, default='12')
    parser.add_argument('--prompts', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results_dynamic_sink')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = [int(x.strip()) for x in args.layers.split(',')]
    prompts = load_prompts(args.prompts, args.num_samples)
    
    if not args.eval_only:
        config = DynamicSinkConfig(
            num_steps=args.num_steps, num_samples=args.num_samples,
            seed=args.seed, top_k=args.top_k,
            intervention_layers=layers, measure_layers=[6, 12, 18],
        )
        run_experiment(config, output_dir, prompts, args.device)
    
    # Evaluate
    print("\n" + "="*60)
    print("CLIP EVALUATION")
    print("="*60)
    
    prompts_file = output_dir / "prompts.txt"
    if prompts_file.exists():
        with open(prompts_file) as f:
            prompts = [l.strip() for l in f if l.strip()]
    
    b_imgs, i_imgs = [], []
    for i in range(len(prompts)):
        bp = output_dir / "images_baseline" / f"{i:03d}.png"
        ip = output_dir / f"images_dynamic_top{args.top_k}" / f"{i:03d}.png"
        if bp.exists() and ip.exists():
            b_imgs.append(Image.open(bp).convert("RGB"))
            i_imgs.append(Image.open(ip).convert("RGB"))
    
    if not b_imgs:
        print("No images found")
        return
    
    print(f"Evaluating {len(b_imgs)} image pairs...")
    b_clips = evaluate_clip(b_imgs, prompts[:len(b_imgs)], args.device)
    i_clips = evaluate_clip(i_imgs, prompts[:len(i_imgs)], args.device)
    stats = compute_paired_stats(b_clips, i_clips)
    
    print(f"\nBaseline:     {stats['baseline_mean']:.4f} ± {stats['baseline_std']:.4f}")
    print(f"Intervention: {stats['intervention_mean']:.4f} ± {stats['intervention_std']:.4f}")
    print(f"Δ:            {stats['delta_mean']:+.4f}")
    print(f"95% CI:       [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}]")
    print(f"p-value:      {stats['p_value']:.4f}")
    
    with open(output_dir / 'clip_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    if stats['ci_lower'] <= 0 <= stats['ci_upper']:
        print("✓ Dynamic sink removal does NOT hurt quality (CI includes 0)")
    else:
        print("⚠ Significant effect detected")
    print("="*60)


if __name__ == "__main__":
    main()
