#!/usr/bin/env python3
"""
E2: Hyperparameter Sensitivity Sweep (4-GPU Parallel Version)

使用论文主实现 DynamicSinkJointAttnProcessor

4 GPU 并行策略：
- 7 个配置分到 4 个 GPU
- GPU 0: CFG=3.0, CFG=7.5
- GPU 1: CFG=12.0, steps=8
- GPU 2: steps=50, euler
- GPU 3: dpm++
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict
import tempfile


CONFIGS = [
    {"cfg": 3.0, "steps": 20, "scheduler": "default"},
    {"cfg": 7.5, "steps": 20, "scheduler": "default"},
    {"cfg": 12.0, "steps": 20, "scheduler": "default"},
    {"cfg": 7.5, "steps": 8, "scheduler": "default"},
    {"cfg": 7.5, "steps": 50, "scheduler": "default"},
    {"cfg": 7.5, "steps": 20, "scheduler": "euler"},
    {"cfg": 7.5, "steps": 20, "scheduler": "dpm++"},
]

# GPU 分配
GPU_ASSIGNMENT = {
    0: [0, 1],      # CFG=3.0, CFG=7.5
    1: [2, 3],      # CFG=12.0, steps=8
    2: [4, 5],      # steps=50, euler
    3: [6],         # dpm++
}


def run_single_config(gpu_id: int, config_indices: List[int], prompts_file: str, 
                      output_dir: str, num_prompts: int) -> str:
    """在单个 GPU 上运行指定配置"""
    
    configs_json = json.dumps([CONFIGS[i] for i in config_indices])
    
    script = f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

import json
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy import stats

# 切换到正确目录
import sys
sys.path.insert(0, "{os.getcwd()}")

from dynamic_sink_processor import DynamicSinkPatcher

def load_prompts(path, n):
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts[:n] if n else prompts

def compute_clip_score(images, prompts, device):
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

def compute_lpips(images1, images2, device):
    import lpips
    loss_fn = lpips.LPIPS(net="alex").to(device)
    
    scores = []
    for img1, img2 in zip(images1, images2):
        t1 = torch.from_numpy(np.array(img1)).permute(2,0,1).float() / 127.5 - 1
        t2 = torch.from_numpy(np.array(img2)).permute(2,0,1).float() / 127.5 - 1
        t1 = t1.unsqueeze(0).to(device)
        t2 = t2.unsqueeze(0).to(device)
        with torch.no_grad():
            d = loss_fn(t1, t2).item()
        scores.append(d)
    return np.array(scores)

def main():
    from diffusers import StableDiffusion3Pipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
    
    configs = {configs_json}
    prompts_file = "{prompts_file}"
    output_dir = Path("{output_dir}")
    num_prompts = {num_prompts}
    gpu_id = {gpu_id}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = load_prompts(prompts_file, num_prompts)
    device = "cuda"
    
    results = []
    
    for cfg_idx, config in enumerate(configs):
        cfg = config["cfg"]
        steps = config["steps"]
        scheduler_name = config["scheduler"]
        
        print(f"\\n[GPU {{gpu_id}}] Config: CFG={{cfg}}, steps={{steps}}, scheduler={{scheduler_name}}")
        
        # Load pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
        )
        
        # Set scheduler
        if scheduler_name == "euler":
            try:
                pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            except:
                results.append({{"skipped": True, **config}})
                del pipe
                torch.cuda.empty_cache()
                continue
        elif scheduler_name == "dpm++":
            try:
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            except:
                results.append({{"skipped": True, **config}})
                del pipe
                torch.cuda.empty_cache()
                continue
        
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        
        # Sanity check
        print(f"  [Sanity] No-op check for scheduler={{scheduler_name}}...")
        gen = torch.Generator(device=device).manual_seed(42)
        baseline_test = pipe("test", num_inference_steps=steps, guidance_scale=cfg, generator=gen).images[0]
        baseline_arr = np.array(baseline_test.convert("RGB"), dtype=np.int16)
        
        patcher = DynamicSinkPatcher(intervention_layers=[12], measure_layers=[12], top_k=1)
        patcher.patch(pipe.transformer)
        patcher.set_intervention_enabled(False)
        
        gen = torch.Generator(device=device).manual_seed(42)
        noop_test = pipe("test", num_inference_steps=steps, guidance_scale=cfg, generator=gen).images[0]
        noop_arr = np.array(noop_test.convert("RGB"), dtype=np.int16)
        
        patcher.unpatch()
        
        diff = int(np.abs(baseline_arr - noop_arr).max())
        if diff != 0:
            print(f"  ✗ FAIL: No-op diff = {{diff}}")
            results.append({{"skipped": True, "sanity_failed": True, **config}})
            del pipe
            torch.cuda.empty_cache()
            continue
        print(f"  ✓ PASS: No-op diff = 0")
        
        # Generate
        baseline_images = []
        intervened_images = []
        
        for i, prompt in enumerate(tqdm(prompts, desc=f"GPU {{gpu_id}} CFG={{cfg}}")):
            seed = 1000 + i
            
            gen = torch.Generator(device=device).manual_seed(seed)
            img_base = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg, generator=gen).images[0]
            baseline_images.append(img_base)
            
            patcher = DynamicSinkPatcher(intervention_layers=[12], measure_layers=[12], top_k=1)
            patcher.patch(pipe.transformer)
            
            gen = torch.Generator(device=device).manual_seed(seed)
            img_int = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg, generator=gen).images[0]
            intervened_images.append(img_int)
            
            patcher.unpatch()
        
        # Metrics
        baseline_scores = compute_clip_score(baseline_images, prompts, device)
        intervened_scores = compute_clip_score(intervened_images, prompts, device)
        lpips_scores = compute_lpips(baseline_images, intervened_images, device)
        
        delta = intervened_scores - baseline_scores
        boot_means = [float(np.mean(delta[np.random.choice(len(delta), len(delta), replace=True)])) for _ in range(1000)]
        
        _, p_value = stats.ttest_rel(baseline_scores, intervened_scores)
        
        results.append({{
            "cfg": cfg,
            "steps": steps,
            "scheduler": scheduler_name,
            "delta_clip_mean": float(np.mean(delta)),
            "delta_clip_ci_low": float(np.percentile(boot_means, 2.5)),
            "delta_clip_ci_high": float(np.percentile(boot_means, 97.5)),
            "p_value": float(p_value),
            "lpips_mean": float(np.mean(lpips_scores)),
        }})
        
        del pipe
        torch.cuda.empty_cache()
    
    # Save results
    result_file = output_dir / f"e2_gpu{{gpu_id}}_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\\n[GPU {{gpu_id}}] Saved: {{result_file}}")

if __name__ == "__main__":
    main()
'''
    
    # 写入临时脚本
    script_file = Path(output_dir) / f"_worker_gpu{gpu_id}.py"
    script_file.parent.mkdir(parents=True, exist_ok=True)
    with open(script_file, "w") as f:
        f.write(script)
    
    return str(script_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_e2_multi_gpu")
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=4)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成并行脚本
    scripts = []
    for gpu_id in range(args.num_gpus):
        if gpu_id in GPU_ASSIGNMENT:
            script = run_single_config(
                gpu_id=gpu_id,
                config_indices=GPU_ASSIGNMENT[gpu_id],
                prompts_file=args.prompts_file,
                output_dir=str(output_dir),
                num_prompts=args.num_prompts,
            )
            scripts.append((gpu_id, script))
    
    # 并行执行
    print(f"Launching {len(scripts)} GPU workers...")
    processes = []
    for gpu_id, script in scripts:
        log_file = output_dir / f"gpu{gpu_id}.log"
        with open(log_file, "w") as log:
            p = subprocess.Popen(
                ["python", script],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append((gpu_id, p, log_file))
            print(f"  GPU {gpu_id}: PID {p.pid}, log: {log_file}")
    
    # 等待完成
    print("\nWaiting for all workers to complete...")
    for gpu_id, p, log_file in processes:
        p.wait()
        status = "✓" if p.returncode == 0 else "✗"
        print(f"  GPU {gpu_id}: {status} (return code: {p.returncode})")
    
    # 合并结果
    print("\nMerging results...")
    all_results = []
    for gpu_id in range(args.num_gpus):
        result_file = output_dir / f"e2_gpu{gpu_id}_results.json"
        if result_file.exists():
            with open(result_file) as f:
                all_results.extend(json.load(f))
    
    # 按配置顺序排序
    config_order = {(c["cfg"], c["steps"], c["scheduler"]): i for i, c in enumerate(CONFIGS)}
    all_results.sort(key=lambda r: config_order.get((r["cfg"], r["steps"], r["scheduler"]), 999))
    
    # 保存合并结果
    final_file = output_dir / "e2_results.json"
    with open(final_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Final results: {final_file}")
    
    # 打印摘要
    print("\n" + "=" * 100)
    print("E2 HYPERPARAMETER SENSITIVITY (4-GPU)")
    print("=" * 100)
    print(f"{'CFG':>6} {'Steps':>6} {'Scheduler':>10} {'ΔCLIP-T':>12} {'95% CI':>24} {'p':>8} {'LPIPS':>8}")
    print("-" * 100)
    
    for r in all_results:
        if r.get("skipped"):
            print(f"{r['cfg']:>6.1f} {r['steps']:>6} {r['scheduler']:>10} {'—':>12} {'—':>24} {'—':>8} {'—':>8}")
        else:
            ci_str = f"[{r['delta_clip_ci_low']:+.4f}, {r['delta_clip_ci_high']:+.4f}]"
            p_str = f"{r['p_value']:.3f}" if r['p_value'] >= 0.001 else "<.001"
            print(f"{r['cfg']:>6.1f} {r['steps']:>6} {r['scheduler']:>10} "
                  f"{r['delta_clip_mean']:>+12.4f} {ci_str:>24} {p_str:>8} {r['lpips_mean']:>8.4f}")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
