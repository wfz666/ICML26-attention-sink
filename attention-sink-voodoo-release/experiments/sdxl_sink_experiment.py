#!/usr/bin/env python3
"""
SDXL Cross-Attention Sink Intervention (Diffusers-Faithful Implementation)

目标：验证 text-sink story 在 SDXL (explicit cross-attention) 上的表现
对比 SD3 (joint attention) 的结果

SDXL 架构特点：
- UNet 结构（不是 DiT）
- Cross-attention: Query=image, Key/Value=text
- 所有 cross-attn keys 都是 text tokens（没有 image keys）
- 因此：SDXL 的 sinks 必然是 text tokens

实现注意：
- 使用 diffusers 官方的 head_to_batch_dim / batch_to_head_dim
- 正确处理 attention_mask (prepare_attention_mask)
- 干预在 logits 上做（softmax 前），与 SD3 实验一致
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy import stats


# ============================================================
# SDXL Cross-Attention Sink Processor (Diffusers-Faithful)
# ============================================================

class SDXLCrossAttnSinkProcessor:
    """
    SDXL Cross-Attention processor with dynamic sink intervention.
    
    Diffusers-faithful implementation:
    - Uses attn.head_to_batch_dim / batch_to_head_dim
    - Uses attn.prepare_attention_mask
    - Uses attn.scale
    - Handles residual_connection and rescale_output_factor
    
    SDXL cross-attention 结构：
    - Query: image features [B, H*W, D]
    - Key/Value: text embeddings [B, 77, D] (CLIP) or [B, 77+?, D] (with pooled)
    
    注意：SDXL 的 cross-attn keys 全部来自 text，
    所以 sink intervention 等价于 text-only intervention。
    """
    
    def __init__(
        self,
        layer_name: str,
        top_k: int = 1,
        intervention_enabled: bool = True,
        mask_value: float = 1e4,
        original_processor = None,
    ):
        self.layer_name = layer_name
        self.top_k = top_k
        self.intervention_enabled = intervention_enabled
        self.mask_value = mask_value
        self.original_processor = original_processor
        
        self.sink_records = []
        
        # For verification: track before/after mass
        self.mass_before = []
        self.mass_after = []
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # If no encoder_hidden_states -> self-attention; defer to original processor
        if encoder_hidden_states is None:
            if self.original_processor is not None:
                return self.original_processor(
                    attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs
                )
            # Fallback: simple self-attention
            return self._simple_attention(attn, hidden_states, hidden_states, attention_mask)
        
        # If intervention disabled, use original processor for exact match
        if not self.intervention_enabled and self.original_processor is not None:
            return self.original_processor(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs
            )
        
        # Cross-attention with sink intervention (diffusers-faithful style)
        batch_size, q_len, _ = hidden_states.shape
        
        # Save residual for potential residual connection
        residual = hidden_states
        
        # Optional norms (diffusers Attention modules may define these)
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        if getattr(attn, "norm_cross", False) and hasattr(attn, "norm_encoder_hidden_states"):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        # Projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Get dimensions
        heads = attn.heads
        inner_dim = key.shape[-1]
        head_dim = inner_dim // heads
        
        # Reshape to [B*H, L, head_dim] using official helpers when available
        if hasattr(attn, "head_to_batch_dim"):
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
        else:
            # Fallback reshape
            query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2).reshape(batch_size * heads, -1, head_dim)
            key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2).reshape(batch_size * heads, -1, head_dim)
            value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2).reshape(batch_size * heads, -1, head_dim)
        
        # Prepare attention mask (critical for correctness)
        attention_mask_prepared = None
        if hasattr(attn, "prepare_attention_mask"):
            attention_mask_prepared = attn.prepare_attention_mask(attention_mask, q_len, batch_size)
            # Normalize mask shape to match logits [B*H, Q, K]
            if attention_mask_prepared is not None:
                # Common shapes:
                #  - [B, 1, 1, K] or [B, 1, K]  -> expand/repeat to [B*H, Q, K]
                #  - [B*H, 1, K] or [B*H, Q, K] -> already head-batched
                if attention_mask_prepared.dim() == 4:
                    # [B, 1, 1, K] -> [B, 1, K]
                    attention_mask_prepared = attention_mask_prepared[:, 0, 0, :]
                if attention_mask_prepared.dim() == 3 and attention_mask_prepared.shape[0] == batch_size:
                    # [B, 1, K] or [B, Q, K] -> repeat heads
                    attention_mask_prepared = attention_mask_prepared.repeat_interleave(heads, dim=0)
                elif attention_mask_prepared.dim() == 2 and attention_mask_prepared.shape[0] == batch_size:
                    # [B, K] -> [B, 1, K] -> repeat heads
                    attention_mask_prepared = attention_mask_prepared[:, None, :].repeat_interleave(heads, dim=0)
        elif attention_mask is not None:
            attention_mask_prepared = attention_mask
        
        # Compute attention logits with correct scaling (diffusers-faithful baddbmm path)
        scale = getattr(attn, "scale", head_dim ** -0.5)
        
        # logits: [B*H, Q, K]
        attn_logits = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], device=query.device, dtype=query.dtype),
            query,
            key.transpose(-1, -2),
            beta=0.0,
            alpha=scale,
        )
        
        # Apply attention mask BEFORE softmax (correct order)
        if attention_mask_prepared is not None:
            attn_logits = attn_logits + attention_mask_prepared
        
        # Baseline probs for sink detection (float32 softmax for stability)
        attn_probs_base = F.softmax(attn_logits.float(), dim=-1)  # [B*H, Q, K]
        
        # Reshape to [B, H, Q, K] for per-head sink detection
        n_key = attn_probs_base.shape[-1]
        attn_probs_4d = attn_probs_base.view(batch_size, heads, attn_probs_base.shape[1], n_key)
        
        # Find dynamic sinks: keys that receive highest average attention mass
        incoming_mass = attn_probs_4d.mean(dim=2)  # [B, H, K]
        topk_values, topk_indices = torch.topk(incoming_mass, k=self.top_k, dim=-1)  # [B, H, k]
        
        # Record sinks (all are text tokens in cross-attention)
        with torch.no_grad():
            for b in range(batch_size):
                for h in range(heads):
                    for k in range(self.top_k):
                        sink_idx = topk_indices[b, h, k].item()
                        sink_mass = topk_values[b, h, k].item()
                        self.sink_records.append({
                            "layer": self.layer_name,
                            "head": h,
                            "sink_idx": sink_idx,
                            "n_key": n_key,
                            "sink_mass": sink_mass,
                        })
                        self.mass_before.append(sink_mass)
        
        # Apply intervention if enabled
        if self.intervention_enabled:
            # Build mask on logits: [B, H, K] -> [B*H, 1, K]
            mask = torch.zeros_like(incoming_mass)  # [B, H, K]
            mask.scatter_(-1, topk_indices, 1.0)
            mask = mask.view(batch_size * heads, 1, n_key)  # [B*H, 1, K]
            
            # Subtract large bias on selected key positions (logit bias intervention)
            attn_logits = attn_logits - (self.mask_value * mask).to(attn_logits.dtype)
            attn_probs = F.softmax(attn_logits.float(), dim=-1)
            
            # Track mass after intervention for verification
            with torch.no_grad():
                attn_probs_4d_after = attn_probs.view(batch_size, heads, -1, n_key)
                incoming_mass_after = attn_probs_4d_after.mean(dim=2)
                for b in range(batch_size):
                    for h in range(heads):
                        for k in range(self.top_k):
                            sink_idx = topk_indices[b, h, k].item()
                            mass_after = incoming_mass_after[b, h, sink_idx].item()
                            self.mass_after.append(mass_after)
        else:
            attn_probs = attn_probs_base
        
        # Compute output: [B*H, Q, head_dim]
        attn_probs = attn_probs.to(value.dtype)
        hidden_states = torch.bmm(attn_probs, value)
        
        # Restore back to [B, Q, inner_dim]
        if hasattr(attn, "batch_to_head_dim"):
            hidden_states = attn.batch_to_head_dim(hidden_states)
        else:
            # Fallback reshape
            hidden_states = hidden_states.view(batch_size, heads, -1, head_dim).transpose(1, 2).reshape(batch_size, -1, inner_dim)
        
        # Output projection + optional dropout
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)
        
        # Handle residual connection if defined
        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        
        # Handle output rescaling if defined
        rescale = getattr(attn, "rescale_output_factor", None)
        if rescale is not None:
            hidden_states = hidden_states / rescale
        
        return hidden_states
    
    def _simple_attention(self, attn, query_states, kv_states, attention_mask):
        """Fallback simple self-attention (fixed: mask before softmax)."""
        batch_size = query_states.shape[0]
        
        query = attn.to_q(query_states)
        key = attn.to_k(kv_states)
        value = attn.to_v(kv_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        scale = head_dim ** -0.5
        logits = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Apply mask BEFORE softmax (correct order)
        if attention_mask is not None:
            logits = logits + attention_mask
        
        attn_weights = F.softmax(logits.float(), dim=-1).to(value.dtype)
        
        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states
    
    def get_intervention_stats(self) -> Dict:
        """Get intervention verification statistics."""
        if not self.mass_before or not self.mass_after:
            return {}
        
        before = np.array(self.mass_before)
        after = np.array(self.mass_after)
        reduction = before / (after + 1e-10)
        
        return {
            "mass_before_mean": float(np.mean(before)),
            "mass_after_mean": float(np.mean(after)),
            "reduction_factor_mean": float(np.mean(reduction)),
            "reduction_factor_min": float(np.min(reduction)),
        }


class SDXLSinkPatcher:
    """Patcher for SDXL cross-attention sink intervention."""
    
    def __init__(
        self,
        target_blocks: List[str] = ["mid"],  # "down", "mid", "up"
        top_k: int = 1,
        mask_value: float = 1e4,
    ):
        self.target_blocks = target_blocks
        self.top_k = top_k
        self.mask_value = mask_value
        
        self.processors: Dict[str, SDXLCrossAttnSinkProcessor] = {}
        self.original_processors: Dict[str, any] = {}
        self._unet = None
    
    def patch(self, unet) -> None:
        """Patch SDXL UNet with sink processors."""
        self._unet = unet
        self.original_processors = unet.attn_processors.copy()
        
        new_processors = {}
        patched_count = 0
        
        for name, original_proc in self.original_processors.items():
            # Check if this is a cross-attention in target blocks
            should_patch = False
            for block in self.target_blocks:
                if block in name and "attn2" in name:  # attn2 is cross-attention
                    should_patch = True
                    break
            
            if should_patch:
                proc = SDXLCrossAttnSinkProcessor(
                    layer_name=name,
                    top_k=self.top_k,
                    intervention_enabled=True,
                    mask_value=self.mask_value,
                    original_processor=original_proc,
                )
                new_processors[name] = proc
                self.processors[name] = proc
                patched_count += 1
            else:
                new_processors[name] = original_proc
        
        unet.set_attn_processor(new_processors)
        print(f"Patched {patched_count} cross-attention layers")
        print(f"  Target blocks: {self.target_blocks}")
        if patched_count > 0:
            print(f"  Patched layers:")
            for layer_name in sorted(self.processors.keys()):
                print(f"    - {layer_name}")
    
    def unpatch(self) -> None:
        """Restore original processors."""
        if self._unet is not None and self.original_processors:
            self._unet.set_attn_processor(self.original_processors)
            print("Restored original processors")
        
        self.processors.clear()
        self.original_processors.clear()
        self._unet = None
    
    def set_intervention_enabled(self, enabled: bool) -> None:
        """Enable/disable intervention."""
        for proc in self.processors.values():
            proc.intervention_enabled = enabled
    
    def get_all_sink_records(self) -> List[Dict]:
        """Get all sink records from all processors."""
        records = []
        for proc in self.processors.values():
            records.extend(proc.sink_records)
        return records
    
    def get_intervention_stats(self) -> Dict:
        """Get aggregated intervention verification stats."""
        all_before = []
        all_after = []
        for proc in self.processors.values():
            all_before.extend(proc.mass_before)
            all_after.extend(proc.mass_after)
        
        if not all_before or not all_after:
            return {}
        
        before = np.array(all_before)
        after = np.array(all_after)
        reduction = before / (after + 1e-10)
        
        return {
            "mass_before_mean": float(np.mean(before)),
            "mass_after_mean": float(np.mean(after)),
            "reduction_factor_mean": float(np.mean(reduction)),
            "reduction_factor_min": float(np.min(reduction)),
            "n_interventions": len(before),
        }
    
    def clear_records(self) -> None:
        """Clear sink records and stats."""
        for proc in self.processors.values():
            proc.sink_records = []
            proc.mass_before = []
            proc.mass_after = []


# ============================================================
# Metrics
# ============================================================

def compute_clip_score(images: List[Image.Image], prompts: List[str], device: str) -> np.ndarray:
    """Compute CLIP-T scores."""
    import open_clip
    
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    
    scores = []
    for img, prompt in zip(images, prompts):
        img_input = preprocess(img).unsqueeze(0).to(device)
        text_input = tokenizer([prompt]).to(device)
        
        with torch.no_grad():
            img_feat = model.encode_image(img_input)
            txt_feat = model.encode_text(text_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            score = (img_feat @ txt_feat.T).item()
        
        scores.append(score)
    
    return np.array(scores)


def compute_lpips(images1: List[Image.Image], images2: List[Image.Image], device: str) -> np.ndarray:
    """Compute LPIPS distances."""
    import lpips
    
    loss_fn = lpips.LPIPS(net="alex").to(device)
    
    scores = []
    for img1, img2 in zip(images1, images2):
        t1 = torch.from_numpy(np.array(img1)).permute(2, 0, 1).float() / 127.5 - 1
        t2 = torch.from_numpy(np.array(img2)).permute(2, 0, 1).float() / 127.5 - 1
        t1 = t1.unsqueeze(0).to(device)
        t2 = t2.unsqueeze(0).to(device)
        
        with torch.no_grad():
            d = loss_fn(t1, t2).item()
        
        scores.append(d)
    
    return np.array(scores)


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, 100 * alpha), np.percentile(boot_means, 100 * (1 - alpha))


# ============================================================
# Main Experiment
# ============================================================

def run_sdxl_sink_experiment(
    prompts_file: str,
    output_dir: str,
    num_prompts: int = 32,
    device: str = "cuda",
):
    """Run SDXL cross-attention sink experiment."""
    from diffusers import StableDiffusionXLPipeline
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    prompts = prompts[:num_prompts]
    print(f"Using {len(prompts)} prompts")
    
    # Load SDXL
    print("\nLoading SDXL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # =========================================================================
    # Sanity Check 1: No-op should produce identical output
    # =========================================================================
    print("\n[Sanity Check 1] No-op verification...")
    
    gen = torch.Generator(device=device).manual_seed(42)
    baseline_test = pipe("a cat", num_inference_steps=20, generator=gen).images[0]
    baseline_arr = np.array(baseline_test.convert("RGB"), dtype=np.int16)
    
    patcher = SDXLSinkPatcher(target_blocks=["mid"], top_k=1)
    patcher.patch(pipe.unet)
    patcher.set_intervention_enabled(False)  # Disable intervention
    
    gen = torch.Generator(device=device).manual_seed(42)
    noop_test = pipe("a cat", num_inference_steps=20, generator=gen).images[0]
    noop_arr = np.array(noop_test.convert("RGB"), dtype=np.int16)
    
    patcher.unpatch()
    
    diff = int(np.abs(baseline_arr - noop_arr).max())
    if diff != 0:
        print(f"  ✗ FAIL: No-op diff = {diff}")
        print("  Sanity check failed! Aborting.")
        return
    print(f"  ✓ PASS: No-op diff = 0")
    
    # =========================================================================
    # Sanity Check 2: Intervention actually reduces sink mass (multi-prompt)
    # =========================================================================
    print("\n[Sanity Check 2] Intervention verification (10 prompts)...")
    
    patcher = SDXLSinkPatcher(target_blocks=["mid"], top_k=1)
    patcher.patch(pipe.unet)
    # intervention_enabled=True by default
    
    # Run on 10 prompts to get robust statistics
    test_prompts = prompts[:10] if len(prompts) >= 10 else prompts
    all_reduction_factors = []
    
    for i, prompt in enumerate(test_prompts):
        patcher.clear_records()
        gen = torch.Generator(device=device).manual_seed(42 + i)
        _ = pipe(prompt, num_inference_steps=20, generator=gen).images[0]
        
        int_stats = patcher.get_intervention_stats()
        if int_stats and int_stats['reduction_factor_mean'] > 0:
            all_reduction_factors.append(int_stats['reduction_factor_mean'])
    
    patcher.unpatch()
    
    if all_reduction_factors:
        rf = np.array(all_reduction_factors)
        print(f"  Reduction factor statistics (n={len(rf)} prompts):")
        print(f"    Median: {np.median(rf):.1f}x")
        print(f"    p5:     {np.percentile(rf, 5):.1f}x")
        print(f"    p95:    {np.percentile(rf, 95):.1f}x")
        print(f"    Min:    {np.min(rf):.1f}x")
        
        # Save for results
        intervention_verification = {
            "n_prompts": len(rf),
            "reduction_factor_median": float(np.median(rf)),
            "reduction_factor_p5": float(np.percentile(rf, 5)),
            "reduction_factor_p95": float(np.percentile(rf, 95)),
            "reduction_factor_min": float(np.min(rf)),
        }
        
        if np.median(rf) > 100:
            print(f"  ✓ PASS: Intervention effectively suppresses sinks")
        else:
            print(f"  ⚠ WARNING: Reduction factor seems low")
    else:
        print(f"  ⚠ WARNING: No intervention stats recorded")
        intervention_verification = None
    
    # =========================================================================
    # Generate baseline and intervened images
    # =========================================================================
    print("\n[Generation] Baseline vs Intervention...")
    
    baseline_images = []
    intervened_images = []
    
    # Patch once, toggle intervention
    patcher = SDXLSinkPatcher(target_blocks=["mid"], top_k=1)
    patcher.patch(pipe.unet)
    
    # Save patched layer names for results
    patched_layers = sorted(list(patcher.processors.keys()))
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        seed = 1000 + i
        
        # Baseline (intervention disabled)
        patcher.set_intervention_enabled(False)
        patcher.clear_records()
        gen = torch.Generator(device=device).manual_seed(seed)
        img_base = pipe(prompt, num_inference_steps=20, generator=gen).images[0]
        baseline_images.append(img_base)
        
        # Intervention (enabled)
        patcher.set_intervention_enabled(True)
        patcher.clear_records()
        gen = torch.Generator(device=device).manual_seed(seed)
        img_int = pipe(prompt, num_inference_steps=20, generator=gen).images[0]
        intervened_images.append(img_int)
    
    patcher.unpatch()
    
    # Save sample images
    (output_path / "samples").mkdir(exist_ok=True)
    for i in range(min(5, len(prompts))):
        baseline_images[i].save(output_path / "samples" / f"{i:03d}_baseline.png")
        intervened_images[i].save(output_path / "samples" / f"{i:03d}_intervened.png")
    
    # =========================================================================
    # Compute metrics
    # =========================================================================
    print("\n[Metrics] Computing CLIP-T and LPIPS...")
    
    baseline_scores = compute_clip_score(baseline_images, prompts, device)
    intervened_scores = compute_clip_score(intervened_images, prompts, device)
    lpips_scores = compute_lpips(baseline_images, intervened_images, device)
    
    delta = intervened_scores - baseline_scores
    ci_low, ci_high = bootstrap_ci(delta)
    _, p_value = stats.ttest_rel(baseline_scores, intervened_scores)
    
    # =========================================================================
    # Results
    # =========================================================================
    results = {
        "model": "SDXL",
        "target_blocks": ["mid"],
        "top_k": 1,
        "num_prompts": len(prompts),
        "patched_layers": patched_layers,
        "delta_clip_mean": float(np.mean(delta)),
        "delta_clip_std": float(np.std(delta)),
        "delta_clip_ci_low": float(ci_low),
        "delta_clip_ci_high": float(ci_high),
        "p_value": float(p_value),
        "lpips_mean": float(np.mean(lpips_scores)),
        "lpips_std": float(np.std(lpips_scores)),
        "baseline_clip_mean": float(np.mean(baseline_scores)),
        "intervened_clip_mean": float(np.mean(intervened_scores)),
        "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
        "intervention_verification": intervention_verification,
    }
    
    # Save results
    with open(output_path / "sdxl_sink_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SDXL CROSS-ATTENTION SINK INTERVENTION")
    print("=" * 70)
    print(f"Target: mid-block cross-attention, top-1 sink")
    print(f"N = {len(prompts)} prompts")
    print("-" * 70)
    print(f"ΔCLIP-T:  {np.mean(delta):+.4f} ± {np.std(delta):.4f}")
    print(f"95% CI:   [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"p-value:  {p_value:.4f}")
    print(f"LPIPS:    {np.mean(lpips_scores):.4f} ± {np.std(lpips_scores):.4f}")
    print("=" * 70)
    
    if results["ci_includes_zero"]:
        print("✓ 95% CI includes zero → No significant change in semantic alignment")
    else:
        print("⚠ 95% CI excludes zero")
    
    print(f"\nResults saved to: {output_path / 'sdxl_sink_results.json'}")
    
    # Generate comparison with SD3
    print("\n" + "=" * 70)
    print("COMPARISON: SD3 vs SDXL")
    print("=" * 70)
    print("Both models show text-sink dominance:")
    print("  SD3:  Joint attention → 99.99% sinks on text tokens")
    print("  SDXL: Cross-attention → 100% sinks are text tokens (by design)")
    print("")
    print("Intervention effect:")
    print(f"  SDXL: ΔCLIP-T = {np.mean(delta):+.4f}, LPIPS = {np.mean(lpips_scores):.4f}")
    print("  (Compare with SD3 E3 text_only results)")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_sdxl_sink")
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    run_sdxl_sink_experiment(
        prompts_file=args.prompts_file,
        output_dir=args.output_dir,
        num_prompts=args.num_prompts,
        device=args.device,
    )


if __name__ == "__main__":
    main()
