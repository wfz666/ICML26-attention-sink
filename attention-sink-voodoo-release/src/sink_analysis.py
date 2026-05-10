"""
Attention Sink Analysis for Diffusion Transformers (v3)
========================================================
使用AttnProcessor API实现干预，兼容SD3/Flux的Joint Attention。

关键改进：
- 不替换attention模块，而是设置自定义AttnProcessor
- 正确处理Joint Attention的tuple返回值
- 支持SD3, SD3.5, Flux等现代diffusion transformers

Usage:
    python sink_analysis.py --model sd3 --steps 50 --num_samples 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict, List, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np


# =============================================================================
# 1. Metrics & Data Structures
# =============================================================================

@dataclass
class SinkMetrics:
    """单步的sink指标"""
    timestep: float  # t/T normalized
    layer_idx: int
    
    # Sink strength metrics
    first_token_attn_ratio: float  # sink token收到的attention比例
    top_k_concentration: float     # top-k tokens的attention集中度
    entropy: float                 # attention分布熵
    
    # Activation metrics  
    max_activation: float
    p95_activation: float
    
    # Optional: per-head breakdown
    per_head_sink_ratio: Optional[torch.Tensor] = None


@dataclass 
class InterventionConfig:
    """干预配置"""
    
    # 干预类型
    intervention_type: Literal["none", "score_only", "value_only", "both"] = "none"
    
    # Score-path干预参数
    score_method: Literal["prob_scale", "logit_scale", "logit_shift"] = "prob_scale"
    score_scale: float = 0.1  # η: 0=完全移除, 1=不变
    
    # Value-path干预参数
    value_method: Literal["zero", "mean", "noise", "lerp"] = "zero"
    value_noise_std: float = 0.1
    value_lerp_alpha: float = 0.5  # α: 0=完全替换为mean, 1=不变
    
    # 干预位置
    intervention_layers: List[int] = field(default_factory=lambda: [12])
    sink_token_indices: List[int] = field(default_factory=lambda: [0])


@dataclass
class ExperimentConfig:
    """实验配置"""
    model_name: str = "sd3"
    num_steps: int = 50
    num_samples: int = 16
    seed: int = 42
    
    # 要测量的层
    measure_layers: List[int] = field(default_factory=lambda: [6, 12, 18])
    
    # 干预配置
    intervention: InterventionConfig = field(default_factory=InterventionConfig)


# =============================================================================
# 2. Custom Attention Processor (核心实现)
# =============================================================================

class SinkAwareAttnProcessor:
    """
    自定义Attention Processor，支持干预和metrics记录。
    
    关键设计：
    - 保存原始processor，joint/cross attention时直接调用原始processor
    - 只在self-attention时使用我们的实现
    - 这样保证与SD3/SD3.5/Flux的完全兼容
    """

    # 类变量：调试计数器
    _self_attn_calls = 0
    _cross_attn_calls = 0

    @classmethod
    def reset_counters(cls):
        cls._self_attn_calls = 0
        cls._cross_attn_calls = 0

    @classmethod
    def print_stats(cls):
        total = cls._self_attn_calls + cls._cross_attn_calls
        if total > 0:
            print(f"  Attention calls: joint/cross={cls._cross_attn_calls}, self={cls._self_attn_calls}")
            print(f"  (Interventions apply to joint attention on intervention_layers)")

    def __init__(
        self,
        layer_idx: int,
        config: ExperimentConfig,
        metrics_buffer: Dict[int, List[SinkMetrics]],
        original_processor = None,  # 保存原始processor
    ):
        self.layer_idx = layer_idx
        self.config = config
        self.metrics_buffer = metrics_buffer
        self.current_timestep: float = 0.0
        self._enabled = True
        self.original_processor = original_processor  # 关键：保存原始processor
        self.block_id = None
        self.processor_name = None

    def set_timestep(self, t: float):
        self.current_timestep = t

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def __call__(
        self,
        attn,  # Attention module
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        自定义attention处理。

        SD3架构说明：
        - SD3使用MMDiT（Multi-Modal DiT），所有attention都是Joint Attention
        - encoder_hidden_states始终不为None
        - 我们需要在Joint Attention内部做干预，而不是跳过它

        策略：
        - 如果intervention_type != "none"且在intervention_layers中：
          使用我们的实现（支持干预）
        - 否则：使用原始processor
        """
        # 检查是否需要干预
        intervention = self.config.intervention
        should_intervene = (
            self._enabled and
            intervention.intervention_type != "none" and
            self.layer_idx in intervention.intervention_layers
        )

        # 检查是否需要记录metrics
        should_measure = (
            self._enabled and
            self.layer_idx in self.config.measure_layers
        )

        # 如果不需要干预也不需要记录，直接使用原始processor
        if not should_intervene and not should_measure:
            if encoder_hidden_states is not None:
                SinkAwareAttnProcessor._cross_attn_calls += 1
            else:
                SinkAwareAttnProcessor._self_attn_calls += 1

            if self.original_processor is not None:
                return self.original_processor(
                    attn, hidden_states, encoder_hidden_states,
                    attention_mask, temb, *args, **kwargs
                )

        # 需要干预或记录：使用我们的实现
        is_joint = encoder_hidden_states is not None

        if is_joint:
            SinkAwareAttnProcessor._cross_attn_calls += 1
            return self._joint_attention_with_intervention(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, should_intervene, should_measure,
                *args, **kwargs
            )
        else:
            SinkAwareAttnProcessor._self_attn_calls += 1
            return self._standard_attention_forward(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, *args, **kwargs
            )

    def _joint_attention_with_intervention(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        should_intervene: bool,
        should_measure: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SD3 Joint Attention with intervention support.

        基于diffusers的JointAttnProcessor2_0实现，在关键位置插入干预。
        """
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # === Norm (如果有的话) ===
        # SD3的attention模块可能已经在block level做了norm

        # === Image stream: Q, K, V ===
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # === Text stream: Q, K, V (使用add projections) ===
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        # === Concatenate for joint attention ===
        # [B, H, N_txt + N_img, D]
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # === Attention computation ===
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 检查attn_weights是否有问题
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            # 打印调试信息（只打印一次）
            if not hasattr(self, '_debug_printed'):
                print(f"  [DEBUG] Layer {self.layer_idx}: attn_weights has NaN/inf!")
                print(f"    attn_weights range: [{attn_weights.min().item():.2f}, {attn_weights.max().item():.2f}]")
                self._debug_printed = True

        attn_probs = F.softmax(attn_weights, dim=-1)

        # === Score-path干预 ===
        if should_intervene:
            intervention = self.config.intervention
            if intervention.intervention_type in ["score_only", "both"]:
                attn_probs = self._apply_prob_scale(attn_probs)

        # === 记录metrics ===
        if should_measure:
            self._record_metrics(attn_probs, hidden_states)

        # === Value-path干预 ===
        if should_intervene:
            intervention = self.config.intervention
            if intervention.intervention_type in ["value_only", "both"]:
                value = self._apply_value_intervention(value)

        # === Attention output ===
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)

        # === Split back to text and image ===
        # concatenate顺序是[text, image]，所以split后第一部分是text，第二部分是image
        encoder_hidden_states_len = encoder_hidden_states.shape[1]
        text_output, image_output = hidden_states.split(
            [encoder_hidden_states_len, hidden_states.shape[1] - encoder_hidden_states_len],
            dim=1
        )

        # === Output projections ===
        # image通过to_out
        image_output = attn.to_out[0](image_output)
        if len(attn.to_out) > 1:
            image_output = attn.to_out[1](image_output)  # Dropout if exists

        # text通过to_add_out
        text_output = attn.to_add_out(text_output)

        # 返回顺序：(hidden_states, encoder_hidden_states) = (image, text)
        # 这与JointAttnProcessor2_0一致
        return image_output, text_output

    def _standard_attention_forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """标准self/cross attention"""

        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        # Attention mask
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # Q, K, V
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Multi-head reshape
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Attention
        attn_probs = self._compute_attention(query, key, head_dim, attention_mask)

        # === 干预 ===
        attn_probs, value = self._apply_interventions(attn_probs, value, hidden_states)

        # Output
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)

        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        return hidden_states

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        head_dim: int,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算attention概率"""
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        return attn_probs

    def _apply_interventions(
        self,
        attn_probs: torch.Tensor,
        value: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用干预并记录metrics"""

        intervention = self.config.intervention

        # 检查是否需要干预
        do_score = (
            intervention.intervention_type in ["score_only", "both"] and
            self.layer_idx in intervention.intervention_layers and
            self._enabled
        )

        do_value = (
            intervention.intervention_type in ["value_only", "both"] and
            self.layer_idx in intervention.intervention_layers and
            self._enabled
        )

        do_measure = self.layer_idx in self.config.measure_layers and self._enabled

        # Score干预
        if do_score and intervention.score_method == "prob_scale":
            attn_probs = self._apply_prob_scale(attn_probs)

        # 记录metrics
        if do_measure:
            self._record_metrics(attn_probs, hidden_states)

        # Value干预
        if do_value:
            value = self._apply_value_intervention(value)

        return attn_probs, value

    def _apply_prob_scale(self, attn_probs: torch.Tensor) -> torch.Tensor:
        """概率缩放（保持rest比例）"""
        eta = float(self.config.intervention.score_scale)
        sink_indices = list(self.config.intervention.sink_token_indices)

        if eta >= 1.0 or len(sink_indices) == 0:
            return attn_probs
        if eta < 0.0:
            eta = 0.0

        attn_probs = attn_probs.clone()
        M = attn_probs.shape[-1]

        sink_mask = torch.zeros(M, device=attn_probs.device, dtype=attn_probs.dtype)
        for idx in sink_indices:
            if 0 <= idx < M:
                sink_mask[idx] = 1.0
        sink_mask = sink_mask.view(1, 1, 1, -1)
        rest_mask = 1.0 - sink_mask

        p_sink_old = (attn_probs * sink_mask).sum(dim=-1, keepdim=True)
        p_rest_old = 1.0 - p_sink_old

        attn_sink_new = attn_probs * sink_mask * eta
        p_sink_new = attn_sink_new.sum(dim=-1, keepdim=True)
        p_rest_new_total = 1.0 - p_sink_new

        eps = 1e-12
        scale = p_rest_new_total / torch.clamp(p_rest_old, min=eps)
        attn_rest_new = (attn_probs * rest_mask) * scale

        attn_new = attn_sink_new + attn_rest_new
        attn_new = attn_new / torch.clamp(attn_new.sum(dim=-1, keepdim=True), min=eps)

        return attn_new

    def _apply_value_intervention(self, value: torch.Tensor) -> torch.Tensor:
        """Value干预"""
        intervention = self.config.intervention
        method = intervention.value_method
        sink_indices = intervention.sink_token_indices

        value = value.clone()

        for sink_idx in sink_indices:
            if sink_idx < 0 or sink_idx >= value.shape[2]:
                continue

            if method == "zero":
                value[:, :, sink_idx, :] = 0.0

            elif method == "mean":
                mean_value = value.mean(dim=2, keepdim=True).squeeze(2)
                value[:, :, sink_idx, :] = mean_value

            elif method == "noise":
                std = intervention.value_noise_std
                noise = torch.randn_like(value[:, :, sink_idx, :]) * std
                value[:, :, sink_idx, :] = noise

            elif method == "lerp":
                alpha = intervention.value_lerp_alpha
                mean_value = value.mean(dim=2, keepdim=True).squeeze(2)
                original = value[:, :, sink_idx, :]
                value[:, :, sink_idx, :] = alpha * original + (1 - alpha) * mean_value

        return value

    def _record_metrics(self, attn_probs: torch.Tensor, hidden_states: torch.Tensor):
        """记录metrics"""
        B, H, N, M = attn_probs.shape
        sink_indices = self.config.intervention.sink_token_indices

        # Sink attention ratio
        sink_attn_total = 0.0
        for sink_idx in sink_indices:
            if 0 <= sink_idx < M:
                sink_attn_total += attn_probs[:, :, :, sink_idx].mean().item()

        # Top-k concentration
        top_k = min(5, M)
        topk_vals, _ = attn_probs.topk(top_k, dim=-1)
        top_k_concentration = topk_vals.sum(dim=-1).mean().item()

        # Entropy (with numerical stability - use float32 to avoid overflow)
        eps = 1e-10
        try:
            # 转换为float32进行计算，避免half precision溢出
            attn_probs_f32 = attn_probs.float()

            # 检查输入
            if torch.isnan(attn_probs_f32).any() or torch.isinf(attn_probs_f32).any():
                entropy = -1.0  # 标记：输入有NaN/inf
            else:
                attn_probs_safe = attn_probs_f32.clamp(min=eps, max=1.0)
                # 归一化确保是有效概率分布
                row_sums = attn_probs_safe.sum(dim=-1, keepdim=True)
                attn_probs_safe = attn_probs_safe / row_sums.clamp(min=eps)

                log_probs = torch.log(attn_probs_safe)
                # 分步计算避免溢出
                entropy_per_token = -(attn_probs_safe * log_probs).sum(dim=-1)  # [B, H, N]
                entropy_raw = entropy_per_token.mean()

                if torch.isnan(entropy_raw) or torch.isinf(entropy_raw):
                    entropy = -2.0  # 标记：计算产生NaN/inf
                else:
                    entropy = entropy_raw.item()
        except Exception as e:
            entropy = -3.0  # 标记：异常

        # Activation stats
        max_activation = hidden_states.abs().max().item()
        p95_activation = torch.quantile(
            hidden_states.abs().flatten().float(), 0.95
        ).item()

        metrics = SinkMetrics(
            timestep=self.current_timestep,
            layer_idx=self.layer_idx,
            first_token_attn_ratio=sink_attn_total,
            top_k_concentration=top_k_concentration,
            entropy=entropy,
            max_activation=max_activation,
            p95_activation=p95_activation,
        )

        self.metrics_buffer[self.layer_idx].append(metrics)


# =============================================================================
# 3. Transformer Patcher (使用AttnProcessor API)
# =============================================================================

class TransformerPatcher:
    """
    使用AttnProcessor API来patch transformer。

    不替换attention模块，而是设置自定义的AttnProcessor。
    这样可以正确处理SD3的Joint Attention。
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.metrics_buffer: Dict[int, List[SinkMetrics]] = defaultdict(list)
        self.processors: Dict[str, SinkAwareAttnProcessor] = {}
        self.original_processors: Dict[str, Any] = {}
        self._transformer = None

    def patch(self, transformer: nn.Module) -> None:
        """
        使用set_attn_processor设置自定义processor

        关键：layer_idx是processor枚举顺序，不一定等于transformer block index！
        我们尝试从processor name解析出真实的block_id。
        """
        self._transformer = transformer

        # 获取原始processors
        self.original_processors = transformer.attn_processors.copy()

        # 解析processor name -> block_id映射
        # 典型name格式: "transformer_blocks.12.attn1.processor" 或 "blocks.5.attn.processor"
        import re

        def parse_block_id(name: str) -> Optional[int]:
            """从processor name解析block_id"""
            # 尝试匹配 "blocks.N" 或 "transformer_blocks.N" 或类似模式
            match = re.search(r'(?:transformer_)?blocks?\.(\d+)', name)
            if match:
                return int(match.group(1))
            return None

        # 创建新的processors，同时建立映射
        new_processors = {}
        self._name_to_layer_idx = {}
        self._layer_idx_to_block_id = {}

        layer_idx = 0
        for name, original_processor in self.original_processors.items():
            block_id = parse_block_id(name)

            custom_processor = SinkAwareAttnProcessor(
                layer_idx=layer_idx,
                config=self.config,
                metrics_buffer=self.metrics_buffer,
                original_processor=original_processor,  # 关键：传入原始processor
            )
            # 存储block_id供后续使用
            custom_processor.block_id = block_id
            custom_processor.processor_name = name

            new_processors[name] = custom_processor
            self.processors[name] = custom_processor
            self._name_to_layer_idx[name] = layer_idx
            self._layer_idx_to_block_id[layer_idx] = block_id

            layer_idx += 1

        # 设置新processors
        transformer.set_attn_processor(new_processors)

        # 打印映射信息
        print(f"\nPatched {len(new_processors)} attention processors")
        print(f"  Measure layers: {self.config.measure_layers}")
        print(f"  Intervention layers: {self.config.intervention.intervention_layers}")

        # 打印前30个processor的映射（帮助调试layer对齐）
        print("\n  Processor name -> layer_idx -> block_id mapping (first 30):")
        for i, (name, proc) in enumerate(list(self.processors.items())[:30]):
            block_str = f"block={proc.block_id}" if proc.block_id is not None else "block=?"
            is_measure = "📊" if i in self.config.measure_layers else "  "
            is_intervene = "🔧" if i in self.config.intervention.intervention_layers else "  "
            print(f"    [{i:02d}] {is_measure}{is_intervene} {block_str:10} {name}")

        if len(new_processors) > 30:
            print(f"    ... and {len(new_processors) - 30} more")

        print("\n  NOTE: SD3 uses Joint Attention for all layers.")
        print("        Interventions now apply INSIDE joint attention (not skipped).")

    def unpatch(self) -> None:
        """恢复原始processors"""
        if self._transformer is None:
            return

        if self.original_processors:
            self._transformer.set_attn_processor(self.original_processors)
            print("Restored original attention processors")

        self.processors.clear()
        self.original_processors.clear()
        self._transformer = None

    def set_timestep(self, t: float) -> None:
        """设置所有processor的timestep"""
        for processor in self.processors.values():
            processor.set_timestep(t)

    def enable(self) -> None:
        """启用干预"""
        for processor in self.processors.values():
            processor.enable()

    def disable(self) -> None:
        """禁用干预"""
        for processor in self.processors.values():
            processor.disable()

    def get_metrics_dataframe(self):
        """导出metrics为DataFrame"""
        import pandas as pd

        records = []
        for layer_idx, metrics_list in self.metrics_buffer.items():
            for m in metrics_list:
                records.append({
                    "timestep": m.timestep,
                    "layer": layer_idx,
                    "sink_ratio": m.first_token_attn_ratio,
                    "top_k_conc": m.top_k_concentration,
                    "entropy": m.entropy,
                    "max_act": m.max_activation,
                    "p95_act": m.p95_activation,
                })
        return pd.DataFrame(records)

    def clear_metrics(self) -> None:
        """清空metrics buffer"""
        self.metrics_buffer.clear()


# =============================================================================
# 4. Visualization
# =============================================================================

def plot_h1_curves(df, save_path: str = "h1_curves.png"):
    """绘制H1曲线"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    layers = sorted(df['layer'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    # Sink ratio
    ax = axes[0, 0]
    for layer, color in zip(layers, colors):
        layer_df = df[df['layer'] == layer]
        grouped = layer_df.groupby('timestep')['sink_ratio'].agg(['mean', 'std'])
        ax.plot(grouped.index, grouped['mean'], color=color, label=f'Layer {layer}')
        ax.fill_between(
            grouped.index,
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            color=color, alpha=0.2
        )
    ax.set_xlabel('Timestep (t/T)')
    ax.set_ylabel('Sink Attention Ratio')
    ax.set_title('H1: Sink Strength vs Timestep')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[0, 1]
    has_entropy_data = False
    for layer, color in zip(layers, colors):
        layer_df = df[df['layer'] == layer]
        grouped = layer_df.groupby('timestep')['entropy'].agg(['mean', 'std'])
        # 过滤掉NaN、0和负值（调试标记）
        valid_mask = (grouped['mean'] > 0) & ~grouped['mean'].isna()
        if valid_mask.any():
            has_entropy_data = True
            ax.plot(grouped.index[valid_mask], grouped['mean'][valid_mask],
                   color=color, label=f'Layer {layer}')
            ax.fill_between(
                grouped.index[valid_mask],
                grouped['mean'][valid_mask] - grouped['std'][valid_mask],
                grouped['mean'][valid_mask] + grouped['std'][valid_mask],
                color=color, alpha=0.2
            )
    if not has_entropy_data:
        # 检查是哪种错误
        mean_entropy = df['entropy'].mean()
        if mean_entropy == -1.0:
            msg = 'Input attn_probs has NaN/inf'
        elif mean_entropy == -2.0:
            msg = 'Entropy calculation produces NaN/inf'
        elif mean_entropy == -3.0:
            msg = 'Exception during calculation'
        elif mean_entropy == 0.0:
            msg = 'All entropy values are 0'
        else:
            msg = f'No valid entropy (mean={mean_entropy:.2f})'
        ax.text(0.5, 0.5, msg, ha='center', va='center',
               transform=ax.transAxes, fontsize=10, color='red')
    ax.set_xlabel('Timestep (t/T)')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Attention Entropy vs Timestep')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-k concentration
    ax = axes[1, 0]
    for layer, color in zip(layers, colors):
        layer_df = df[df['layer'] == layer]
        grouped = layer_df.groupby('timestep')['top_k_conc'].agg(['mean', 'std'])
        ax.plot(grouped.index, grouped['mean'], color=color, label=f'Layer {layer}')
    ax.set_xlabel('Timestep (t/T)')
    ax.set_ylabel('Top-5 Concentration')
    ax.set_title('Attention Concentration vs Timestep')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Max activation
    ax = axes[1, 1]
    for layer, color in zip(layers, colors):
        layer_df = df[df['layer'] == layer]
        grouped = layer_df.groupby('timestep')['max_act'].agg(['mean', 'std'])
        ax.plot(grouped.index, grouped['mean'], color=color, label=f'Layer {layer}')
    ax.set_xlabel('Timestep (t/T)')
    ax.set_ylabel('Max Activation')
    ax.set_title('Activation Magnitude vs Timestep')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved H1 curves to {save_path}")


def plot_intervention_sweep(sweep_values, quality_scores, sweep_type: str, save_path: str):
    """绘制干预强度sweep曲线"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(sweep_values, quality_scores, 'o-', linewidth=2, markersize=8)

    if sweep_type == "score":
        ax.set_xlabel('Score Scale (η)')
        ax.set_title('Quality vs Score Intervention Strength')
    else:
        ax.set_xlabel('Value Lerp Alpha')
        ax.set_title('Quality vs Value Intervention Strength')

    ax.set_ylabel('Quality Score (CLIP/FID)')
    ax.grid(True, alpha=0.3)

    # 标记baseline (η=1.0或α=1.0)
    baseline_idx = -1
    for i, v in enumerate(sweep_values):
        if abs(v - 1.0) < 0.01:
            baseline_idx = i
            break

    if baseline_idx >= 0:
        ax.axhline(y=quality_scores[baseline_idx], color='red', linestyle='--',
                   alpha=0.5, label='Baseline (no intervention)')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sweep curve to {save_path}")