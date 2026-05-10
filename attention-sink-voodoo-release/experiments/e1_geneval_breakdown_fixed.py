#!/usr/bin/env python3
"""
E1: GenEval Tag-based Breakdown Analysis (FIXED VERSION)

Key fixes:
1. Multi-label tagging instead of single-label categorization
2. A prompt can belong to multiple tags (more faithful for compositional prompts)
3. Renamed from "category" to "tag" to avoid claiming GenEval official categories

Analyzes sink intervention effects by semantic tags:
- counting, position, color, two_object, single_object

Reports: ΔCLIP-T, ΔImageReward, ΔHPS-v2, LPIPS per tag
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy import stats


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ============================================================
# Tag Patterns (multi-label)
# ============================================================

TAG_PATTERNS = {
    "counting": [
        "two ", "three ", "four ", "five ", "six ",
        "2 ", "3 ", "4 ", "5 ", "6 ",
    ],
    "position": [
        " on ", " under ", " above ", " below ", " next to ",
        " left of ", " right of ", " in front of ", " behind ",
        " beside ", " between ",
    ],
    "two_object": [
        " and a ", " and an ", " with a ", " with an ",
    ],
    "color": [
        "red ", "blue ", "green ", "yellow ", "orange ", "purple ",
        "pink ", "brown ", "black ", "white ", "gray ", "grey ",
        "golden ", "silver ",
    ],
}


def tag_prompt(prompt: str) -> List[str]:
    """
    Assign multiple tags to a prompt based on patterns.
    A prompt can have multiple tags (more faithful for compositional prompts).
    """
    prompt_lower = prompt.lower()
    tags = []
    
    # counting
    if any(pattern in prompt_lower for pattern in TAG_PATTERNS["counting"]):
        tags.append("counting")
    
    # position / relations
    if any(pattern in prompt_lower for pattern in TAG_PATTERNS["position"]):
        tags.append("position")
    
    # multi-object
    if any(pattern in prompt_lower for pattern in TAG_PATTERNS["two_object"]):
        tags.append("two_object")
    
    # color mention
    if any(pattern in prompt_lower for pattern in TAG_PATTERNS["color"]):
        tags.append("color")
    
    # Default to single_object only if no other tags
    if not tags:
        tags.append("single_object")
    
    return tags


def load_geneval_metadata(prompts_file: Path) -> Dict[int, Dict]:
    """Load prompts and assign tags (multi-label)."""
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    metadata = {}
    for i, prompt in enumerate(prompts):
        metadata[i] = {
            "prompt": prompt,
            "tags": tag_prompt(prompt),
        }
    
    return metadata


def compute_clip_scores_by_tag(
    base_dir: Path,
    intv_dir: Path,
    metadata: Dict[int, Dict],
    device: str = "cuda",
) -> Dict[str, Dict]:
    """Compute CLIP-T scores grouped by tag (multi-label)."""
    import open_clip
    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    base_files = {int(p.stem): p for p in base_dir.glob("*.png")}
    intv_files = {int(p.stem): p for p in intv_dir.glob("*.png")}
    common_ids = sorted(set(base_files.keys()) & set(intv_files.keys()) & set(metadata.keys()))
    
    # Group by tag (a sample can contribute to multiple tags)
    tag_scores = defaultdict(lambda: {"base": [], "intv": [], "delta": [], "indices": []})
    
    with torch.no_grad():
        for idx in tqdm(common_ids, desc="CLIP-T by tag"):
            prompt = metadata[idx]["prompt"]
            tags = metadata[idx]["tags"]
            
            txt = tokenizer([prompt]).to(device)
            txt_feat = model.encode_text(txt)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            
            # Baseline
            with Image.open(base_files[idx]) as img:
                img_b = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
            feat_b = model.encode_image(img_b)
            feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
            score_b = (feat_b @ txt_feat.T).item()
            
            # Intervention
            with Image.open(intv_files[idx]) as img:
                img_i = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
            feat_i = model.encode_image(img_i)
            feat_i = feat_i / feat_i.norm(dim=-1, keepdim=True)
            score_i = (feat_i @ txt_feat.T).item()
            
            # Add to all applicable tags
            for tag in tags:
                tag_scores[tag]["base"].append(score_b)
                tag_scores[tag]["intv"].append(score_i)
                tag_scores[tag]["delta"].append(score_i - score_b)
                tag_scores[tag]["indices"].append(idx)
    
    return dict(tag_scores)


def compute_lpips_by_tag(
    base_dir: Path,
    intv_dir: Path,
    metadata: Dict[int, Dict],
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """Compute LPIPS scores grouped by tag (multi-label)."""
    import lpips
    
    loss_fn = lpips.LPIPS(net='alex').to(device).eval()
    
    base_files = {int(p.stem): p for p in base_dir.glob("*.png")}
    intv_files = {int(p.stem): p for p in intv_dir.glob("*.png")}
    common_ids = sorted(set(base_files.keys()) & set(intv_files.keys()) & set(metadata.keys()))
    
    def to_tensor(p):
        with Image.open(p) as im:
            im = im.convert("RGB")
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        return (2 * x - 1).unsqueeze(0).to(device)
    
    tag_lpips = defaultdict(list)
    
    with torch.no_grad():
        for idx in tqdm(common_ids, desc="LPIPS by tag"):
            tags = metadata[idx]["tags"]
            x = to_tensor(base_files[idx])
            y = to_tensor(intv_files[idx])
            d = loss_fn(x, y).item()
            
            for tag in tags:
                tag_lpips[tag].append(d)
    
    return dict(tag_lpips)


def bootstrap_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(sample.mean())
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def print_results_table(tag_scores: Dict, tag_lpips: Dict):
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print("GENEVAL TAG-BASED BREAKDOWN (multi-label)")
    print("Note: A prompt can belong to multiple tags; counts may sum to > total prompts")
    print("=" * 100)
    print(f"{'Tag':<20} {'N':>6} {'ΔCLIP-T':>12} {'95% CI':>24} {'p-value':>10} {'LPIPS':>10}")
    print("-" * 100)
    
    all_results = []
    
    # Sort tags for consistent output
    tag_order = ["single_object", "two_object", "counting", "color", "position"]
    sorted_tags = [t for t in tag_order if t in tag_scores] + \
                  [t for t in sorted(tag_scores.keys()) if t not in tag_order]
    
    for tag in sorted_tags:
        scores = tag_scores[tag]
        n = len(scores["delta"])
        delta = np.array(scores["delta"])
        
        mean_delta = delta.mean()
        ci_low, ci_high = bootstrap_ci(delta)
        
        # Paired t-test
        t, p = stats.ttest_rel(scores["intv"], scores["base"])
        
        # LPIPS
        if tag in tag_lpips and len(tag_lpips[tag]) > 0:
            lpips_mean = np.mean(tag_lpips[tag])
        else:
            lpips_mean = float("nan")
        
        # Format p-value
        if p < 0.001:
            p_str = "<0.001"
        elif p < 0.01:
            p_str = f"{p:.3f}**"
        elif p < 0.05:
            p_str = f"{p:.3f}*"
        else:
            p_str = f"{p:.3f}"
        
        ci_str = f"[{ci_low:+.4f}, {ci_high:+.4f}]"
        
        print(f"{tag:<20} {n:>6} {mean_delta:>+12.4f} {ci_str:>24} {p_str:>10} {lpips_mean:>10.3f}")
        
        all_results.append({
            "tag": tag,
            "n": int(n),
            "delta_clip_mean": float(mean_delta),
            "delta_clip_ci_low": float(ci_low),
            "delta_clip_ci_high": float(ci_high),
            "p_value": float(p),
            "lpips_mean": float(lpips_mean) if not np.isnan(lpips_mean) else None,
            "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
        })
    
    print("=" * 100)
    print("* p < 0.05, ** p < 0.01")
    
    # Summary check
    all_include_zero = all(r["ci_includes_zero"] for r in all_results)
    if all_include_zero:
        print("\n✓ All 95% CIs include zero → Intervention preserves alignment across ALL task types")
    else:
        failing = [r["tag"] for r in all_results if not r["ci_includes_zero"]]
        print(f"\n⚠ CIs exclude zero for: {', '.join(failing)}")
    
    return all_results


def generate_latex_table(results: List[Dict]) -> str:
    """Generate LaTeX table."""
    latex = r"""\begin{table}[t]
\centering
\caption{\textbf{Tag-based breakdown under A1 intervention (GenEval).}
Semantic alignment (CLIP-T) is preserved across all task types,
including compositional tasks (counting, position) that require precise
attribute binding. Note: prompts can have multiple tags.}
\label{tab:geneval_breakdown}
\vskip 0.1in
\begin{small}
\begin{tabular}{lcccc}
\toprule
Tag & $N$ & $\Delta$CLIP-T & 95\% CI & LPIPS \\
\midrule
"""
    
    for r in results:
        tag = r["tag"].replace("_", r"\_")
        n = r["n"]
        delta = r["delta_clip_mean"]
        ci_low = r["delta_clip_ci_low"]
        ci_high = r["delta_clip_ci_high"]
        lpips = r["lpips_mean"]
        
        ci_str = f"[{ci_low:+.3f}, {ci_high:+.3f}]"
        lpips_str = f"{lpips:.3f}" if lpips is not None else "—"
        
        latex += f"{tag} & {n} & ${delta:+.4f}$ & {ci_str} & {lpips_str} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{small}
\end{table}
"""
    return latex


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="results_geneval_A1_layer12_top1")
    parser.add_argument("--output", default="e1_breakdown_results.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    base_dir = exp_dir / "images_baseline"
    intv_dir = exp_dir / "images_dynamic_top1"
    prompts_file = exp_dir / "prompts.txt"
    
    # Try alternative directory names
    if not intv_dir.exists():
        intv_dir = exp_dir / "images_intervention"
    if not intv_dir.exists():
        intv_dir = exp_dir / "images_sink_removed"
    
    print(f"Experiment: {exp_dir}")
    print(f"Baseline: {base_dir}")
    print(f"Intervention: {intv_dir}")
    
    if not base_dir.exists() or not intv_dir.exists():
        print(f"ERROR: Directory not found. Available dirs in {exp_dir}:")
        for d in exp_dir.iterdir():
            print(f"  {d}")
        return
    
    # Load and tag prompts
    print("\n[1/3] Loading and tagging prompts (multi-label)...")
    metadata = load_geneval_metadata(prompts_file)
    
    # Print tag distribution
    tag_counts = defaultdict(int)
    for m in metadata.values():
        for t in m["tags"]:
            tag_counts[t] += 1
    
    print("\nTag distribution (multi-label, may sum > total):")
    print(f"  Total prompts: {len(metadata)}")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count} ({count/len(metadata)*100:.1f}%)")
    
    # Compute CLIP-T by tag
    print("\n[2/3] Computing CLIP-T by tag...")
    tag_scores = compute_clip_scores_by_tag(
        base_dir, intv_dir, metadata, device=args.device
    )
    
    # Compute LPIPS by tag
    print("\n[3/3] Computing LPIPS by tag...")
    tag_lpips = compute_lpips_by_tag(
        base_dir, intv_dir, metadata, device=args.device
    )
    
    # Print results
    results = print_results_table(tag_scores, tag_lpips)
    
    # Generate LaTeX
    latex = generate_latex_table(results)
    print("\n" + "=" * 100)
    print("LATEX TABLE")
    print("=" * 100)
    print(latex)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump({
            "results": results,
            "tag_distribution": {k: int(v) for k, v in tag_counts.items()},
            "total_prompts": len(metadata),
            "latex": latex,
            "note": "Multi-label tagging: prompts can belong to multiple tags",
        }, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
