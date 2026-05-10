#!/usr/bin/env python
"""
Collect H1 Dynamics with Dynamic Sink Definition
=================================================
记录每个 timestep 的 maximum incoming attention mass，
用于替换原来基于 index-0 的 sink ratio。

Output: h1_dynamic_metrics.csv
Columns: timestep, layer, max_incoming_mass, entropy, top_k_conc, max_act, index0_mass, dynamic_sink_position
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse

from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import Attention


class DynamicSinkMeasurementProcessor:
    """测量 dynamic sink metrics，不做干预"""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.current_step = 0
        self.metrics_per_step = defaultdict(list)

    def set_step(self, step: int):
        self.current_step = step

    def clear(self):
        self.metrics_per_step = defaultdict(list)

    def get_dataframe(self) -> pd.DataFrame:
        records = []
        for step, metrics_list in self.metrics_per_step.items():
            for m in metrics_list:
                records.append({
                    'timestep': step,
                    'layer': self.layer_idx,
                    **m
                })
        return pd.DataFrame(records)

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            *args, **kwargs,
    ):
        batch_size = hidden_states.shape[0]

        # === Projections ===
        query_img = attn.to_q(hidden_states)
        key_img = attn.to_k(hidden_states)
        value_img = attn.to_v(hidden_states)

        inner_dim = key_img.shape[-1]
        head_dim = inner_dim // attn.heads

        if encoder_hidden_states is not None:
            query_txt = attn.add_q_proj(encoder_hidden_states)
            key_txt = attn.add_k_proj(encoder_hidden_states)
            value_txt = attn.add_v_proj(encoder_hidden_states)
            key = torch.cat([key_img, key_txt], dim=1)
            value = torch.cat([value_img, value_txt], dim=1)
        else:
            query_txt = None
            key = key_img
            value = value_img

        # Reshape
        query_img = query_img.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if query_txt is not None:
            query_txt = query_txt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Normalization
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            query_img = attn.norm_q(query_img)
        if hasattr(attn, 'norm_k') and attn.norm_k is not None:
            key = attn.norm_k(key)
        if query_txt is not None and hasattr(attn, 'norm_added_q') and attn.norm_added_q is not None:
            query_txt = attn.norm_added_q(query_txt)

        # Attention computation
        scale = head_dim ** -0.5
        attn_logits = torch.matmul(query_img, key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_logits.float(), dim=-1)

        # === Compute metrics ===
        with torch.no_grad():
            # Incoming mass for each key position
            incoming_mass = attn_weights.mean(dim=2)  # [B, H, K]

            # Dynamic sink: max incoming mass
            max_mass_per_head = incoming_mass.max(dim=-1).values  # [B, H]
            max_incoming_mass = max_mass_per_head.mean().item()

            # Position of dynamic sink
            dynamic_sink_pos = incoming_mass.argmax(dim=-1).float().mean().item()

            # Index-0 mass (for comparison)
            index0_mass = incoming_mass[:, :, 0].mean().item()

            # Entropy
            entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(dim=-1).mean().item()

            # Top-5 concentration
            top5_vals, _ = torch.topk(attn_weights, k=5, dim=-1)
            top_k_conc = top5_vals.sum(dim=-1).mean().item()

            # Max activation (use attention output norm as proxy)
            attn_output = torch.matmul(attn_weights.to(value.dtype), value)
            max_act = attn_output.abs().max().item()

            self.metrics_per_step[self.current_step].append({
                'max_incoming_mass': max_incoming_mass,
                'index0_mass': index0_mass,
                'dynamic_sink_position': dynamic_sink_pos,
                'entropy': entropy,
                'top_k_conc': top_k_conc,
                'max_act': max_act,
            })

        # === Normal forward pass (no intervention) ===
        attn_weights = attn_weights.to(value.dtype)
        hidden_states_out = torch.matmul(attn_weights, value)
        hidden_states_out = hidden_states_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        if query_txt is not None:
            attn_logits_txt = torch.matmul(query_txt, key.transpose(-2, -1)) * scale
            attn_weights_txt = F.softmax(attn_logits_txt.float(), dim=-1).to(value.dtype)
            encoder_out = torch.matmul(attn_weights_txt, value)
            encoder_out = encoder_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        else:
            encoder_out = None

        # Output projections
        if hasattr(attn, 'to_out') and attn.to_out is not None:
            if isinstance(attn.to_out, torch.nn.ModuleList):
                for module in attn.to_out:
                    hidden_states_out = module(hidden_states_out)
            else:
                hidden_states_out = attn.to_out(hidden_states_out)

        if encoder_out is not None and hasattr(attn, 'to_add_out') and attn.to_add_out is not None:
            encoder_out = attn.to_add_out(encoder_out)
            return hidden_states_out, encoder_out

        return hidden_states_out, encoder_hidden_states


def collect_h1_dynamics(
        output_dir: Path,
        num_samples: int = 8,
        num_steps: int = 20,
        seed: int = 42,
        layers: list = [6, 12, 18],
        device: str = "cuda",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SD3...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to(device)

    # Install measurement processors
    processors = {}
    original_processors = {}

    for layer_idx in layers:
        block = pipe.transformer.transformer_blocks[layer_idx]
        original_processors[layer_idx] = block.attn.processor
        processors[layer_idx] = DynamicSinkMeasurementProcessor(layer_idx)
        block.attn.processor = processors[layer_idx]

    prompts = [
                  "A photo of a cat sitting on a windowsill",
                  "A beautiful sunset over the ocean",
                  "A professional portrait of a businessman",
                  "An abstract painting with vibrant colors",
                  "A cozy living room with modern furniture",
                  "A mountain landscape with snow-capped peaks",
                  "A delicious plate of pasta with tomato sauce",
                  "A futuristic city skyline at night",
              ][:num_samples]

    print(f"Collecting H1 dynamics for {num_samples} samples, {num_steps} steps...")

    for i, prompt in enumerate(tqdm(prompts, desc="Samples")):
        # Clear metrics
        for p in processors.values():
            p.clear()

        gen = torch.Generator(device=device).manual_seed(seed + i)

        def step_callback(pipe, step, timestep, callback_kwargs):
            normalized_step = step / max(num_steps - 1, 1)
            for p in processors.values():
                p.set_step(normalized_step)
            return callback_kwargs

        _ = pipe(
            prompt,
            num_inference_steps=num_steps,
            generator=gen,
            callback_on_step_end=step_callback,
            output_type="pil",
        )

    # Restore original processors
    for layer_idx, proc in original_processors.items():
        pipe.transformer.transformer_blocks[layer_idx].attn.processor = proc

    # Combine all data
    all_dfs = []
    for layer_idx, proc in processors.items():
        df = proc.get_dataframe()
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Aggregate by timestep and layer
    agg_df = combined_df.groupby(['timestep', 'layer']).agg({
        'max_incoming_mass': 'mean',
        'index0_mass': 'mean',
        'dynamic_sink_position': 'mean',
        'entropy': 'mean',
        'top_k_conc': 'mean',
        'max_act': 'mean',
    }).reset_index()

    # Save
    csv_path = output_dir / 'h1_dynamic_metrics.csv'
    agg_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("H1 DYNAMICS SUMMARY (Dynamic Sink Definition)")
    print("=" * 60)
    for layer in layers:
        layer_df = agg_df[agg_df['layer'] == layer]
        print(f"\nLayer {layer}:")
        print(
            f"  Max incoming mass: {layer_df['max_incoming_mass'].mean() * 100:.2f}% (range: {layer_df['max_incoming_mass'].min() * 100:.2f}% - {layer_df['max_incoming_mass'].max() * 100:.2f}%)")
        print(f"  Index-0 mass:      {layer_df['index0_mass'].mean() * 100:.4f}%")
        print(f"  Dynamic sink pos:  {layer_df['dynamic_sink_position'].mean():.1f}")
        print(f"  Entropy:           {layer_df['entropy'].mean():.2f}")

    return agg_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results_h1_dynamic')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    collect_h1_dynamics(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        seed=args.seed,
    )