#!/usr/bin/env python3
"""
E3: Text vs Image Token Attribution (4-GPU Parallel Version)

4 GPU 并行策略：
- GPU 0: Attribution analysis (sink 统计)
- GPU 1: Ablation mode="none" + mode="text_only"
- GPU 2: Ablation mode="image_only"
- GPU 3: Ablation mode="all"
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict
import numpy as np


def generate_attribution_script(gpu_id: int, prompts_file: str, output_dir: str, 
                                 num_prompts: int) -> str:
    """生成 attribution 分析脚本"""
    
    script = f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import sys
sys.path.insert(0, "{os.getcwd()}")

from dynamic_sink_processor import DynamicSinkPatcher

def load_prompts(path, n):
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts[:n] if n else prompts

def main():
    from diffusers import StableDiffusion3Pipeline
    
    prompts_file = "{prompts_file}"
    output_dir = Path("{output_dir}")
    num_prompts = {num_prompts}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = load_prompts(prompts_file, num_prompts)
    device = "cuda"
    
    print(f"[GPU {gpu_id}] Loading SD3 for attribution analysis...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Sink tracker
    sink_records = []
    
    def track_sink(layer, head, sink_idx, is_text, n_img, n_txt):
        sink_records.append({{
            "layer": layer,
            "head": head,
            "sink_idx": sink_idx,
            "is_text": is_text,
            "n_img": n_img,
            "n_txt": n_txt,
        }})
    
    # Measure-only mode
    patcher = DynamicSinkPatcher(
        intervention_layers=[],  # No intervention
        measure_layers=list(range(24)),  # Measure all layers
        top_k=1,
    )
    patcher.patch(pipe.transformer)
    
    # Set tracker
    for name, proc in pipe.transformer.attn_processors.items():
        if hasattr(proc, 'sink_tracker'):
            proc.sink_tracker = track_sink
    
    print(f"[GPU {gpu_id}] Running attribution on {{len(prompts)}} prompts...")
    for i, prompt in enumerate(tqdm(prompts, desc="Attribution")):
        seed = 1000 + i
        gen = torch.Generator(device=device).manual_seed(seed)
        _ = pipe(prompt, num_inference_steps=20, guidance_scale=7.5, generator=gen).images[0]
    
    patcher.unpatch()
    
    # Analyze - only count records where n_txt > 0 (valid joint attention)
    valid_records = [r for r in sink_records if r.get("n_txt", 0) > 0]
    filtered_out = len(sink_records) - len(valid_records)
    
    text_count = sum(1 for r in valid_records if r["is_text"])
    image_count = sum(1 for r in valid_records if not r["is_text"])
    total = len(valid_records)
    
    # Per-layer breakdown (only valid records)
    layer_stats = defaultdict(lambda: {{"text": 0, "image": 0}})
    for r in valid_records:
        if r["is_text"]:
            layer_stats[r["layer"]]["text"] += 1
        else:
            layer_stats[r["layer"]]["image"] += 1
    
    results = {{
        "total_sinks": total,
        "text_sinks": text_count,
        "image_sinks": image_count,
        "text_ratio": text_count / total if total > 0 else 0,
        "image_ratio": image_count / total if total > 0 else 0,
        "filtered_out": filtered_out,
        "per_layer": dict(layer_stats),
        "num_prompts": len(prompts),
    }}
    
    result_file = output_dir / "attribution_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n[GPU {gpu_id}] Attribution Results:")
    print(f"  Valid records (n_txt>0): {{total}}")
    print(f"  Filtered out (n_txt==0): {{filtered_out}}")
    print(f"  Text sinks:  {{text_count}} ({{100*text_count/total:.1f}}%)" if total > 0 else "  Text sinks: 0")
    print(f"  Image sinks: {{image_count}} ({{100*image_count/total:.1f}}%)" if total > 0 else "  Image sinks: 0")
    print(f"  Saved: {{result_file}}")
    
    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
'''
    
    script_file = Path(output_dir) / f"_worker_attribution.py"
    script_file.parent.mkdir(parents=True, exist_ok=True)
    with open(script_file, "w") as f:
        f.write(script)
    
    return str(script_file)


def generate_ablation_script(gpu_id: int, modes: List[str], prompts_file: str, 
                              output_dir: str, num_prompts: int) -> str:
    """生成 ablation 脚本"""
    
    modes_json = json.dumps(modes)
    
    script = f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy import stats

import sys
sys.path.insert(0, "{os.getcwd()}")

from dynamic_sink_processor import SelectiveSinkPatcher

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
    from diffusers import StableDiffusion3Pipeline
    
    modes = {modes_json}
    prompts_file = "{prompts_file}"
    output_dir = Path("{output_dir}")
    num_prompts = {num_prompts}
    gpu_id = {gpu_id}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = load_prompts(prompts_file, num_prompts)
    device = "cuda"
    
    print(f"[GPU {{gpu_id}}] Loading SD3 for modes: {{modes}}")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Generate baseline first
    print(f"[GPU {{gpu_id}}] Generating baseline...")
    baseline_images = []
    for i, prompt in enumerate(tqdm(prompts, desc="Baseline")):
        seed = 1000 + i
        gen = torch.Generator(device=device).manual_seed(seed)
        img = pipe(prompt, num_inference_steps=20, guidance_scale=7.5, generator=gen).images[0]
        baseline_images.append(img)
    
    baseline_scores = compute_clip_score(baseline_images, prompts, device)
    
    results = []
    
    for mode in modes:
        print(f"\\n[GPU {{gpu_id}}] Mode: {{mode}}")
        
        # Sanity check for mode="none"
        if mode == "none":
            print(f"  [Sanity] Checking no-op...")
            gen = torch.Generator(device=device).manual_seed(42)
            baseline_test = pipe("test", num_inference_steps=20, guidance_scale=7.5, generator=gen).images[0]
            baseline_arr = np.array(baseline_test.convert("RGB"), dtype=np.int16)
            
            patcher = SelectiveSinkPatcher(target_layers=[12], top_k=1, mode="none")
            patcher.patch(pipe.transformer)
            
            gen = torch.Generator(device=device).manual_seed(42)
            noop_test = pipe("test", num_inference_steps=20, guidance_scale=7.5, generator=gen).images[0]
            noop_arr = np.array(noop_test.convert("RGB"), dtype=np.int16)
            
            patcher.unpatch()
            
            diff = int(np.abs(baseline_arr - noop_arr).max())
            if diff != 0:
                print(f"  ✗ FAIL: No-op diff = {{diff}}")
                raise RuntimeError("No-op sanity check failed!")
            print(f"  ✓ PASS: No-op diff = 0")
        
        # Generate with mode
        patcher = SelectiveSinkPatcher(target_layers=[12], top_k=1, mode=mode)
        patcher.patch(pipe.transformer)
        
        mode_images = []
        for i, prompt in enumerate(tqdm(prompts, desc=f"Mode={{mode}}")):
            seed = 1000 + i
            gen = torch.Generator(device=device).manual_seed(seed)
            img = pipe(prompt, num_inference_steps=20, guidance_scale=7.5, generator=gen).images[0]
            mode_images.append(img)
        
        patcher.unpatch()
        
        # Metrics
        mode_scores = compute_clip_score(mode_images, prompts, device)
        lpips_scores = compute_lpips(baseline_images, mode_images, device)
        
        delta = mode_scores - baseline_scores
        boot_means = [float(np.mean(delta[np.random.choice(len(delta), len(delta), replace=True)])) for _ in range(1000)]
        
        results.append({{
            "mode": mode,
            "delta_clip_mean": float(np.mean(delta)),
            "delta_clip_ci_low": float(np.percentile(boot_means, 2.5)),
            "delta_clip_ci_high": float(np.percentile(boot_means, 97.5)),
            "lpips_mean": float(np.mean(lpips_scores)),
            "lpips_std": float(np.std(lpips_scores)),
        }})
        
        print(f"  ΔCLIP-T: {{np.mean(delta):+.4f}}")
        print(f"  LPIPS: {{np.mean(lpips_scores):.4f}}")
    
    # Save
    result_file = output_dir / f"ablation_gpu{{gpu_id}}_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\\n[GPU {{gpu_id}}] Saved: {{result_file}}")
    
    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
'''
    
    script_file = Path(output_dir) / f"_worker_ablation_gpu{gpu_id}.py"
    script_file.parent.mkdir(parents=True, exist_ok=True)
    with open(script_file, "w") as f:
        f.write(script)
    
    return str(script_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_e3_multi_gpu")
    parser.add_argument("--num_prompts_attribution", type=int, default=50)
    parser.add_argument("--num_prompts_ablation", type=int, default=32)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU 分配
    # GPU 0: Attribution
    # GPU 1: none + text_only
    # GPU 2: image_only
    # GPU 3: all
    
    scripts = []
    
    # Attribution (GPU 0)
    attr_script = generate_attribution_script(
        gpu_id=0,
        prompts_file=args.prompts_file,
        output_dir=str(output_dir),
        num_prompts=args.num_prompts_attribution,
    )
    scripts.append((0, attr_script, "attribution"))
    
    # Ablation modes
    mode_assignment = {
        1: ["none", "text_only"],
        2: ["image_only"],
        3: ["all"],
    }
    
    for gpu_id, modes in mode_assignment.items():
        abl_script = generate_ablation_script(
            gpu_id=gpu_id,
            modes=modes,
            prompts_file=args.prompts_file,
            output_dir=str(output_dir),
            num_prompts=args.num_prompts_ablation,
        )
        scripts.append((gpu_id, abl_script, f"ablation_{modes}"))
    
    # 并行执行
    print(f"Launching {len(scripts)} GPU workers...")
    processes = []
    for gpu_id, script, desc in scripts:
        log_file = output_dir / f"gpu{gpu_id}.log"
        with open(log_file, "w") as log:
            p = subprocess.Popen(
                ["python", script],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append((gpu_id, p, log_file, desc))
            print(f"  GPU {gpu_id}: {desc}, PID {p.pid}")
    
    # 等待完成
    print("\nWaiting for all workers...")
    for gpu_id, p, log_file, desc in processes:
        p.wait()
        status = "✓" if p.returncode == 0 else "✗"
        print(f"  GPU {gpu_id} ({desc}): {status}")
    
    # 合并结果
    print("\nMerging results...")
    
    # Attribution
    attr_file = output_dir / "attribution_results.json"
    attr_results = None
    if attr_file.exists():
        with open(attr_file) as f:
            attr_results = json.load(f)
        print(f"\nAttribution Summary (only n_txt>0 records):")
        print(f"  Valid records: {attr_results['total_sinks']}")
        if attr_results.get('filtered_out', 0) > 0:
            print(f"  Filtered out (n_txt==0): {attr_results['filtered_out']}")
        print(f"  Text sinks:  {attr_results['text_sinks']} ({100*attr_results['text_ratio']:.1f}%)")
        print(f"  Image sinks: {attr_results['image_sinks']} ({100*attr_results['image_ratio']:.1f}%)")
    
    # Ablation
    all_ablation = []
    for gpu_id in [1, 2, 3]:
        result_file = output_dir / f"ablation_gpu{gpu_id}_results.json"
        if result_file.exists():
            with open(result_file) as f:
                all_ablation.extend(json.load(f))
    
    # 排序
    mode_order = {"none": 0, "text_only": 1, "image_only": 2, "all": 3}
    all_ablation.sort(key=lambda r: mode_order.get(r["mode"], 99))
    
    # 保存合并结果
    final_results = {
        "attribution": attr_results if attr_file.exists() else None,
        "ablation": all_ablation,
    }
    
    final_file = output_dir / "e3_results.json"
    with open(final_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nFinal results: {final_file}")
    
    # 打印 ablation 摘要
    print("\n" + "=" * 80)
    print("E3 SELECTIVE ABLATION (4-GPU)")
    print("=" * 80)
    print(f"{'Mode':<15} {'ΔCLIP-T':>12} {'95% CI':>28} {'LPIPS':>10}")
    print("-" * 80)
    
    for r in all_ablation:
        ci_str = f"[{r['delta_clip_ci_low']:+.4f}, {r['delta_clip_ci_high']:+.4f}]"
        print(f"{r['mode']:<15} {r['delta_clip_mean']:>+12.4f} {ci_str:>28} {r['lpips_mean']:>10.4f}")
    
    print("=" * 80)
    
    # 关键洞察
    img_lpips = next((r['lpips_mean'] for r in all_ablation if r['mode'] == 'image_only'), 0)
    txt_lpips = next((r['lpips_mean'] for r in all_ablation if r['mode'] == 'text_only'), 0)
    
    if img_lpips > txt_lpips:
        print("\n→ Image-side sinks have LARGER perceptual impact")
    else:
        print("\n→ Text-side sinks have LARGER perceptual impact")


if __name__ == "__main__":
    main()
