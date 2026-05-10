#!/usr/bin/env python
"""
Dynamic Sink Processor for SD3 (论文级实现)
============================================

从 run_dynamic_sink.py 抽取的通用模块。

SD3 Joint Attention 结构:
  Image: query_img @ key.T → [B, H, 4096, 4429]
  Text:  query_txt @ key.T → [B, H, 333, 4429]
  其中 key = concat([key_img, key_txt])

动态 Sink 干预:
  1. 计算 incoming_mass = softmax(QK^T).mean(dim=query)
  2. 找 top-k positions with highest incoming mass
  3. 在 logits 上加 bias（-1e4）mask 这些位置
  4. 重新 softmax

Usage:
    from dynamic_sink_processor import (
        DynamicSinkJointAttnProcessor,
        DynamicSinkPatcher,
    )
    
    patcher = DynamicSinkPatcher(intervention_layers=[12], top_k=1)
    patcher.patch(pipe.transformer)
    
    # 生成图像
    image = pipe("prompt", ...).images[0]
    
    # 恢复
    patcher.unpatch()
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from diffusers.models.attention_processor import Attention


@dataclass
class DynamicSinkConfig:
    """Dynamic sink 配置"""
    top_k: int = 1
    mask_value: float = 1e4
    intervention_layers: List[int] = field(default_factory=lambda: [12])
    measure_layers: List[int] = field(default_factory=lambda: [6, 12, 18])


class DynamicSinkJointAttnProcessor:
    """
    SD3 JointAttnProcessor2_0 的动态 sink 干预版本。
    
    关键设计:
    - Query 分开: img_query [B, 4096, D], txt_query [B, 333, D]
    - K/V 共享: key/value [B, 4429, D] = concat([img, txt])
    - 动态 sink 在 key 维度 (4429) 上检测和干预
    - 干预方式: logit bias（不是 prob scale）
    
    这是论文主实验使用的实现。
    """
    
    def __init__(
        self,
        layer_idx: int,
        top_k: int = 1,
        intervention_enabled: bool = True,
        measure_only: bool = False,
        mask_value: float = 1e4,
        original_processor = None,  # 保存原始 processor 用于 no-op
    ):
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.intervention_enabled = intervention_enabled
        self.measure_only = measure_only
        self.mask_value = mask_value
        self.original_processor = original_processor  # 关键：用于 intervention_enabled=False
        
        self.metrics = defaultdict(list)
        self.current_timestep = 0.0
        
        # For tracking sink positions (E3)
        self.sink_tracker = None
    
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
        
        这是论文主实验的核心实现。
        """
        
        # 关键：如果 intervention_enabled=False 且有原始 processor，直接调用原始实现
        # 这确保 no-op sanity check 能通过（避免数值精度差异）
        if not self.intervention_enabled and self.original_processor is not None:
            return self.original_processor(
                attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs
            )
        
        batch_size = hidden_states.shape[0]
        
        # ===== Image Query =====
        query_img = attn.to_q(hidden_states)  # [B, N_img, D]
        
        # ===== Image Key/Value =====
        key_img = attn.to_k(hidden_states)    # [B, N_img, D]
        value_img = attn.to_v(hidden_states)  # [B, N_img, D]
        
        inner_dim = key_img.shape[-1]
        head_dim = inner_dim // attn.heads
        
        # ===== Text Query/Key/Value (SD3 specific) =====
        if encoder_hidden_states is not None:
            query_txt = attn.add_q_proj(encoder_hidden_states)  # [B, N_txt, D]
            key_txt = attn.add_k_proj(encoder_hidden_states)    # [B, N_txt, D]
            value_txt = attn.add_v_proj(encoder_hidden_states)  # [B, N_txt, D]
            
            # Concatenate K and V (shared between both attention paths)
            # 顺序: [img, txt]
            key = torch.cat([key_img, key_txt], dim=1)      # [B, N_img+N_txt, D]
            value = torch.cat([value_img, value_txt], dim=1)  # [B, N_img+N_txt, D]
            
            n_img = hidden_states.shape[1]
            n_txt = encoder_hidden_states.shape[1]
        else:
            query_txt = None
            key = key_img
            value = value_img
            n_img = hidden_states.shape[1]
            n_txt = 0
        
        # ===== Reshape for multi-head attention =====
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
        if query_txt is not None and hasattr(attn, 'norm_added_q') and attn.norm_added_q is not None:
            query_txt = attn.norm_added_q(query_txt)
        
        # ===== Compute image attention logits =====
        scale = head_dim ** -0.5
        attn_logits_img = torch.matmul(query_img, key.transpose(-2, -1)) * scale
        # attn_logits_img: [B, H, N_img, N_img+N_txt]
        
        # ===== Apply attention mask if provided =====
        # diffusers may pass masks in various shapes; broadcast to [B, H, Q, K].
        if attention_mask is not None:
            m = attention_mask
            if m.dtype != attn_logits_img.dtype:
                m = m.to(attn_logits_img.dtype)
            # Common shapes:
            #  [B, Q, K] -> [B, 1, Q, K]
            #  [B, K]    -> [B, 1, 1, K]
            #  [B, 1, 1, K] / [B, 1, Q, K] already OK
            if m.dim() == 3:
                m = m.unsqueeze(1)
            elif m.dim() == 2:
                m = m.unsqueeze(1).unsqueeze(1)
            attn_logits_img = attn_logits_img + m
        
        # ===== Compute baseline attention weights (for dynamic sink detection) =====
        attn_logits_img_f32 = attn_logits_img.float()
        attn_weights_img_baseline = F.softmax(attn_logits_img_f32, dim=-1)
        
        # ===== Compute incoming mass for dynamic sink detection =====
        # incoming_mass[j] = mean over queries of A[:, :, i, j]
        incoming_mass = attn_weights_img_baseline.mean(dim=2)  # [B, H, N_key]
        
        # ===== Find dynamic top-k sinks (in KEY dimension) =====
        topk_values, topk_indices = torch.topk(incoming_mass, k=self.top_k, dim=-1)
        
        # ===== Track sink positions (for E3 analysis) =====
        if self.sink_tracker is not None:
            with torch.no_grad():
                for b in range(batch_size):
                    for h in range(attn.heads):
                        for k in range(self.top_k):
                            sink_idx = topk_indices[b, h, k].item()
                            # 判断是 image token 还是 text token
                            # key 顺序是 [img, txt]，所以 sink_idx < n_img 是 image
                            is_text = sink_idx >= n_img
                            self.sink_tracker(
                                layer=self.layer_idx,
                                head=h,
                                sink_idx=sink_idx,
                                is_text=is_text,
                                n_img=n_img,
                                n_txt=n_txt,
                            )
        
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
            mask = torch.zeros_like(incoming_mass)  # [B, H, N_key]
            mask.scatter_(-1, topk_indices, 1.0)
            mask = mask.unsqueeze(2)  # [B, H, 1, N_key] - broadcast over query dim
            
            # Apply logit bias to image attention
            attn_logits_img_f32 = attn_logits_img_f32 - self.mask_value * mask
            attn_weights_img = F.softmax(attn_logits_img_f32, dim=-1)
            
            # Record post-intervention metrics
            with torch.no_grad():
                incoming_mass_after = attn_weights_img.mean(dim=2)
                top1_mass_after = incoming_mass_after.max(dim=-1).values.mean().item()
                self.metrics['top1_incoming_mass_after'].append(top1_mass_after)
                
                if top1_mass_after > 0:
                    reduction = top1_mass / max(top1_mass_after, 1e-10)
                    self.metrics['reduction_factor'].append(reduction)
        else:
            attn_weights_img = attn_weights_img_baseline
        
        # ===== Compute image attention output =====
        attn_weights_img = attn_weights_img.to(value.dtype)
        hidden_states_img = torch.matmul(attn_weights_img, value)
        hidden_states_img = hidden_states_img.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        
        # ===== Compute text attention (no intervention) =====
        if query_txt is not None:
            attn_logits_txt = torch.matmul(query_txt, key.transpose(-2, -1)) * scale
            # Apply attention mask to text attention as well
            if attention_mask is not None:
                m = attention_mask
                if m.dtype != attn_logits_txt.dtype:
                    m = m.to(attn_logits_txt.dtype)
                if m.dim() == 3:
                    m = m.unsqueeze(1)
                elif m.dim() == 2:
                    m = m.unsqueeze(1).unsqueeze(1)
                attn_logits_txt = attn_logits_txt + m
            attn_weights_txt = F.softmax(attn_logits_txt.float(), dim=-1).to(value.dtype)
            hidden_states_txt = torch.matmul(attn_weights_txt, value)
            hidden_states_txt = hidden_states_txt.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        else:
            hidden_states_txt = None
        
        # ===== Output projections =====
        hidden_states_img = attn.to_out[0](hidden_states_img)
        if len(attn.to_out) > 1:
            hidden_states_img = attn.to_out[1](hidden_states_img)
        
        if hidden_states_txt is not None:
            hidden_states_txt = attn.to_add_out(hidden_states_txt)
            return hidden_states_img, hidden_states_txt
        
        return hidden_states_img


class DynamicSinkPatcher:
    """
    Transformer patcher using DynamicSinkJointAttnProcessor.
    
    用法:
        patcher = DynamicSinkPatcher(intervention_layers=[12], top_k=1)
        patcher.patch(pipe.transformer)
        
        # ... 生成图像 ...
        
        patcher.unpatch()
    """
    
    def __init__(
        self,
        intervention_layers: List[int] = None,
        measure_layers: List[int] = None,
        top_k: int = 1,
        mask_value: float = 1e4,
        measure_only: bool = False,
    ):
        self.intervention_layers = intervention_layers or [12]
        self.measure_layers = measure_layers or [6, 12, 18]
        self.top_k = top_k
        self.mask_value = mask_value
        self.measure_only = measure_only
        
        self.processors: Dict[int, DynamicSinkJointAttnProcessor] = {}
        self.original_processors: Dict[str, any] = {}
        self._transformer = None
    
    def patch(self, transformer) -> None:
        """Patch transformer with dynamic sink processors."""
        self._transformer = transformer
        
        # Save original processors
        self.original_processors = transformer.attn_processors.copy()
        
        # Create new processors
        new_processors = {}
        
        for name, original_proc in self.original_processors.items():
            # Parse layer index from name
            # Format: "transformer_blocks.12.attn.processor"
            layer_idx = self._parse_layer_idx(name)
            
            if layer_idx is None:
                new_processors[name] = original_proc
                continue
            
            # Check if this layer needs intervention or measurement
            needs_intervention = layer_idx in self.intervention_layers
            needs_measure = layer_idx in self.measure_layers
            
            if needs_intervention or needs_measure:
                proc = DynamicSinkJointAttnProcessor(
                    layer_idx=layer_idx,
                    top_k=self.top_k,
                    intervention_enabled=needs_intervention,
                    measure_only=self.measure_only or not needs_intervention,
                    mask_value=self.mask_value,
                    original_processor=original_proc,  # 传入原始 processor
                )
                new_processors[name] = proc
                self.processors[layer_idx] = proc
            else:
                new_processors[name] = original_proc
        
        transformer.set_attn_processor(new_processors)
        
        print(f"Patched {len(self.processors)} layers")
        print(f"  Intervention: {self.intervention_layers}")
        print(f"  Measure: {self.measure_layers}")
    
    def _parse_layer_idx(self, name: str) -> Optional[int]:
        """Parse layer index from processor name."""
        import re
        match = re.search(r'transformer_blocks\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return None
    
    def unpatch(self) -> None:
        """Restore original processors."""
        if self._transformer is not None and self.original_processors:
            self._transformer.set_attn_processor(self.original_processors)
            print("Restored original processors")
        
        self.processors.clear()
        self.original_processors.clear()
        self._transformer = None
    
    def set_timestep(self, t: float) -> None:
        """Set timestep for all processors."""
        for proc in self.processors.values():
            proc.set_timestep(t)
    
    def set_intervention_enabled(self, enabled: bool) -> None:
        """Enable/disable intervention on all processors."""
        for proc in self.processors.values():
            proc.intervention_enabled = enabled
    
    def set_measure_only(self, measure_only: bool) -> None:
        """Set measure_only mode on all processors."""
        for proc in self.processors.values():
            proc.measure_only = measure_only
    
    def clear_metrics(self) -> None:
        """Clear metrics on all processors."""
        for proc in self.processors.values():
            proc.clear_metrics()
    
    def get_all_metrics(self) -> Dict[int, Dict]:
        """Get metrics from all processors."""
        return {
            layer_idx: proc.get_metrics_summary()
            for layer_idx, proc in self.processors.items()
        }
    
    def set_sink_tracker(self, tracker) -> None:
        """Set sink tracker for E3 analysis."""
        for proc in self.processors.values():
            proc.sink_tracker = tracker


# ============================================================
# Selective intervention support (for E3 text vs image)
# ============================================================

class SelectiveDynamicSinkProcessor(DynamicSinkJointAttnProcessor):
    """
    DynamicSinkJointAttnProcessor with selective masking.
    
    Supports:
    - mode="all": mask all dynamic sinks
    - mode="text_only": mask only sinks in text token positions
    - mode="image_only": mask only sinks in image token positions
    - mode="none": no masking (but still track)
    """
    
    def __init__(
        self,
        layer_idx: int,
        top_k: int = 1,
        mode: str = "all",  # "all", "text_only", "image_only", "none"
        mask_value: float = 1e4,
        original_processor = None,  # 保存原始 processor 用于 no-op
    ):
        super().__init__(
            layer_idx=layer_idx,
            top_k=top_k,
            intervention_enabled=(mode != "none"),
            measure_only=False,
            mask_value=mask_value,
            original_processor=original_processor,
        )
        self.mode = mode
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selective dynamic sink intervention."""
        
        # 关键：如果 mode="none" 且有原始 processor，直接调用原始实现
        if self.mode == "none" and self.original_processor is not None:
            return self.original_processor(
                attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs
            )
        
        batch_size = hidden_states.shape[0]
        
        # ===== Projections (same as parent) =====
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
            
            n_img = hidden_states.shape[1]
            n_txt = encoder_hidden_states.shape[1]
        else:
            query_txt = None
            key = key_img
            value = value_img
            n_img = hidden_states.shape[1]
            n_txt = 0
        
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
        
        # Attention logits
        scale = head_dim ** -0.5
        attn_logits_img = torch.matmul(query_img, key.transpose(-2, -1)) * scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            m = attention_mask
            if m.dtype != attn_logits_img.dtype:
                m = m.to(attn_logits_img.dtype)
            if m.dim() == 3:
                m = m.unsqueeze(1)
            elif m.dim() == 2:
                m = m.unsqueeze(1).unsqueeze(1)
            attn_logits_img = attn_logits_img + m
        
        attn_logits_img_f32 = attn_logits_img.float()
        attn_weights_img_baseline = F.softmax(attn_logits_img_f32, dim=-1)
        
        # Find dynamic sinks
        incoming_mass = attn_weights_img_baseline.mean(dim=2)
        topk_values, topk_indices = torch.topk(incoming_mass, k=self.top_k, dim=-1)
        
        # Track sinks
        if self.sink_tracker is not None:
            with torch.no_grad():
                for b in range(batch_size):
                    for h in range(attn.heads):
                        for k in range(self.top_k):
                            sink_idx = topk_indices[b, h, k].item()
                            is_text = sink_idx >= n_img
                            self.sink_tracker(
                                layer=self.layer_idx,
                                head=h,
                                sink_idx=sink_idx,
                                is_text=is_text,
                                n_img=n_img,
                                n_txt=n_txt,
                            )
        
        # ===== Selective intervention =====
        # Text-vs-image selection is only meaningful when joint keys include text tokens.
        # If n_txt == 0, fall back to "all" (or no-op for text_only/image_only).
        effective_mode = self.mode
        if n_txt == 0 and self.mode in ("text_only", "image_only"):
            effective_mode = "all"
        
        if effective_mode != "none":
            # Create selective mask
            mask = torch.zeros_like(incoming_mass)  # [B, H, N_key]
            
            for b in range(batch_size):
                for h in range(attn.heads):
                    for k in range(self.top_k):
                        sink_idx = topk_indices[b, h, k].item()
                        is_text = sink_idx >= n_img
                        
                        should_mask = False
                        if effective_mode == "all":
                            should_mask = True
                        elif effective_mode == "text_only" and is_text:
                            should_mask = True
                        elif effective_mode == "image_only" and not is_text:
                            should_mask = True
                        
                        if should_mask:
                            mask[b, h, sink_idx] = 1.0
            
            mask = mask.unsqueeze(2)  # [B, H, 1, N_key]
            attn_logits_img_f32 = attn_logits_img_f32 - self.mask_value * mask
            attn_weights_img = F.softmax(attn_logits_img_f32, dim=-1)
        else:
            attn_weights_img = attn_weights_img_baseline
        
        # Output
        attn_weights_img = attn_weights_img.to(value.dtype)
        hidden_states_img = torch.matmul(attn_weights_img, value)
        hidden_states_img = hidden_states_img.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        
        if query_txt is not None:
            attn_logits_txt = torch.matmul(query_txt, key.transpose(-2, -1)) * scale
            # Apply attention mask to text attention
            if attention_mask is not None:
                m = attention_mask
                if m.dtype != attn_logits_txt.dtype:
                    m = m.to(attn_logits_txt.dtype)
                if m.dim() == 3:
                    m = m.unsqueeze(1)
                elif m.dim() == 2:
                    m = m.unsqueeze(1).unsqueeze(1)
                attn_logits_txt = attn_logits_txt + m
            attn_weights_txt = F.softmax(attn_logits_txt.float(), dim=-1).to(value.dtype)
            hidden_states_txt = torch.matmul(attn_weights_txt, value)
            hidden_states_txt = hidden_states_txt.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        else:
            hidden_states_txt = None
        
        # Output projections
        hidden_states_img = attn.to_out[0](hidden_states_img)
        if len(attn.to_out) > 1:
            hidden_states_img = attn.to_out[1](hidden_states_img)
        
        if hidden_states_txt is not None:
            hidden_states_txt = attn.to_add_out(hidden_states_txt)
            return hidden_states_img, hidden_states_txt
        
        return hidden_states_img


class SelectiveSinkPatcher:
    """Patcher for selective sink intervention (E3)."""
    
    def __init__(
        self,
        target_layers: List[int],
        top_k: int = 1,
        mode: str = "all",
        mask_value: float = 1e4,
    ):
        self.target_layers = target_layers
        self.top_k = top_k
        self.mode = mode
        self.mask_value = mask_value
        
        self.processors: Dict[int, SelectiveDynamicSinkProcessor] = {}
        self.original_processors: Dict[str, any] = {}
        self._transformer = None
    
    def patch(self, transformer) -> None:
        """Patch transformer."""
        self._transformer = transformer
        self.original_processors = transformer.attn_processors.copy()
        
        new_processors = {}
        
        for name, original_proc in self.original_processors.items():
            layer_idx = self._parse_layer_idx(name)
            
            if layer_idx is not None and layer_idx in self.target_layers:
                proc = SelectiveDynamicSinkProcessor(
                    layer_idx=layer_idx,
                    top_k=self.top_k,
                    mode=self.mode,
                    mask_value=self.mask_value,
                    original_processor=original_proc,  # 传入原始 processor
                )
                new_processors[name] = proc
                self.processors[layer_idx] = proc
            else:
                new_processors[name] = original_proc
        
        transformer.set_attn_processor(new_processors)
    
    def _parse_layer_idx(self, name: str) -> Optional[int]:
        import re
        match = re.search(r'transformer_blocks\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return None
    
    def unpatch(self) -> None:
        if self._transformer is not None and self.original_processors:
            self._transformer.set_attn_processor(self.original_processors)
        self.processors.clear()
        self.original_processors.clear()
        self._transformer = None
    
    def set_sink_tracker(self, tracker) -> None:
        for proc in self.processors.values():
            proc.sink_tracker = tracker
