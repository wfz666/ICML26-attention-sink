#!/usr/bin/env python3
"""
最小单元测试：验证 dynamic_sink_processor.py 的正确性

测试项目：
1. no-op pixel diff == 0
2. intervention_enabled=False 与 baseline 一致
3. patch/unpatch 后 attn_processors 完整恢复

运行时间：~10 秒（使用 1 step 快速验证）
"""

import torch
import numpy as np
from PIL import Image


def test_processor_correctness():
    """快速测试 processor 正确性"""
    from diffusers import StableDiffusion3Pipeline
    from dynamic_sink_processor import DynamicSinkPatcher, SelectiveSinkPatcher
    
    print("Loading SD3 (this may take a moment)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    
    prompt = "a cat"
    seed = 42
    num_steps = 2  # 1 step 下 intervention 可能不改变输出；用 2 steps 更稳健
    
    # =========================================================================
    # Test 1: Baseline generation
    # =========================================================================
    print("\n[Test 1] Generating baseline...")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    baseline = pipe(prompt, num_inference_steps=num_steps, generator=gen).images[0]
    baseline_arr = np.array(baseline.convert("RGB"), dtype=np.int16)
    print("  ✓ Baseline generated")
    
    # Capture the true-original processor state right after pipeline load
    original_keys_0 = set(pipe.transformer.attn_processors.keys())
    original_types_0 = {k: type(v).__name__ for k, v in pipe.transformer.attn_processors.items()}
    
    # =========================================================================
    # Test 2: DynamicSinkPatcher with intervention_enabled=False
    # =========================================================================
    print("\n[Test 2] DynamicSinkPatcher (intervention_enabled=False)...")
    
    patcher = DynamicSinkPatcher(
        intervention_layers=[12],
        measure_layers=[12],
        top_k=1,
    )
    patcher.patch(pipe.transformer)
    patcher.set_intervention_enabled(False)  # Disable intervention
    
    gen = torch.Generator(device="cuda").manual_seed(seed)
    noop = pipe(prompt, num_inference_steps=num_steps, generator=gen).images[0]
    noop_arr = np.array(noop.convert("RGB"), dtype=np.int16)
    
    patcher.unpatch()
    
    diff = int(np.abs(baseline_arr - noop_arr).max())
    if diff == 0:
        print(f"  ✓ PASS: max|pixel diff| = {diff}")
    else:
        print(f"  ✗ FAIL: max|pixel diff| = {diff}")
        return False
    
    # =========================================================================
    # Test 3: SelectiveSinkPatcher with mode="none"
    # =========================================================================
    print("\n[Test 3] SelectiveSinkPatcher (mode='none')...")
    
    patcher = SelectiveSinkPatcher(
        target_layers=[12],
        top_k=1,
        mode="none",
    )
    patcher.patch(pipe.transformer)
    
    gen = torch.Generator(device="cuda").manual_seed(seed)
    noop2 = pipe(prompt, num_inference_steps=num_steps, generator=gen).images[0]
    noop2_arr = np.array(noop2.convert("RGB"), dtype=np.int16)
    
    patcher.unpatch()
    
    diff = int(np.abs(baseline_arr - noop2_arr).max())
    if diff == 0:
        print(f"  ✓ PASS: max|pixel diff| = {diff}")
    else:
        print(f"  ✗ FAIL: max|pixel diff| = {diff}")
        return False
    
    # =========================================================================
    # Test 4: Verify unpatch restores original processors
    # =========================================================================
    print("\n[Test 4] Verify unpatch restores original processors...")
    
    patcher = DynamicSinkPatcher(intervention_layers=[12], measure_layers=[12], top_k=1)
    patcher.patch(pipe.transformer)
    patcher.unpatch()
    
    restored_keys = set(pipe.transformer.attn_processors.keys())
    restored_types = {k: type(v).__name__ for k, v in pipe.transformer.attn_processors.items()}
    
    if original_keys_0 == restored_keys and original_types_0 == restored_types:
        print("  ✓ PASS: Processors fully restored")
    else:
        print("  ✗ FAIL: Processor restoration mismatch")
        missing = list(original_keys_0 - restored_keys)[:5]
        extra = list(restored_keys - original_keys_0)[:5]
        print(f"    missing keys (sample): {missing}")
        print(f"    extra keys (sample): {extra}")
        return False
    
    # =========================================================================
    # Test 5: Verify intervention actually changes output
    # =========================================================================
    print("\n[Test 5] Verify intervention changes output...")
    
    patcher = DynamicSinkPatcher(
        intervention_layers=[12],
        measure_layers=[12],
        top_k=1,
    )
    patcher.patch(pipe.transformer)
    # intervention_enabled defaults to True
    
    gen = torch.Generator(device="cuda").manual_seed(seed)
    intervened = pipe(prompt, num_inference_steps=num_steps, generator=gen).images[0]
    intervened_arr = np.array(intervened.convert("RGB"), dtype=np.int16)
    
    patcher.unpatch()
    
    diff = int(np.abs(baseline_arr - intervened_arr).max())
    if diff > 0:
        print(f"  ✓ PASS: Intervention changes output (max diff = {diff})")
    else:
        # With num_steps=2, identical output is unlikely; treat as failure
        print("  ✗ FAIL: Intervention produced identical output. Check intervention_layers.")
        return False
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_processor_correctness()
    exit(0 if success else 1)
