#!/usr/bin/env python3
"""
Complete evaluation script for all experiments.
Computes: CLIP-T Δ, p-value, LPIPS, FID shift
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy import stats
from scipy.linalg import sqrtm

# ============================================================
# Configuration
# ============================================================

EXPERIMENTS = {
    # GenEval experiments
    "GenEval_A1": {
        "base_dir": "results_geneval_A1_layer12_top1/images_baseline",
        "intv_dir": "results_geneval_A1_layer12_top1/images_dynamic_top1",
        "prompts_file": "results_geneval_A1_layer12_top1/prompts.txt",
        "description": "Layer 12, Top-1 sink removal",
    },
    "GenEval_A2": {
        "base_dir": "results_geneval_A2_multilayer_top1/images_baseline",
        "intv_dir": "results_geneval_A2_multilayer_top1/images_dynamic_top1",
        "prompts_file": "results_geneval_A2_multilayer_top1/prompts.txt",
        "description": "Multi-layer, Top-1 sink removal",
    },
    "GenEval_A3": {
        "base_dir": "results_geneval_A3_layer12_top5/images_baseline",
        "intv_dir": "results_geneval_A3_layer12_top5/images_dynamic_top5",
        "prompts_file": "results_geneval_A3_layer12_top5/prompts.txt",
        "description": "Layer 12, Top-5 sink removal",
    },
    # COCO experiment
    "COCO_5k_A1": {
        "base_dir": "results_coco_fid/images_baseline",
        "intv_dir": "results_coco_fid/images_A1_top1_L12",
        "prompts_file": "results_coco_fid/prompts.txt",
        "description": "COCO 5k, Layer 12, Top-1 sink removal",
    },
}

# Limit samples for speed (set to None for all)
MAX_SAMPLES_CLIP = 1000
MAX_SAMPLES_LPIPS = 500
MAX_SAMPLES_FID = None  # Use all for FID


# ============================================================
# Metric Functions
# ============================================================

def load_prompts(prompts_file: Path) -> List[str]:
    """Load prompts from file."""
    with open(prompts_file) as f:
        return [line.strip() for line in f if line.strip()]


def get_aligned_files(base_dir: Path, intv_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Get aligned file lists between baseline and intervention."""
    base_files = {p.stem: p for p in base_dir.glob("*.png")}
    intv_files = {p.stem: p for p in intv_dir.glob("*.png")}

    common_keys = sorted(set(base_files.keys()) & set(intv_files.keys()))

    base_aligned = [base_files[k] for k in common_keys]
    intv_aligned = [intv_files[k] for k in common_keys]

    return base_aligned, intv_aligned


def compute_clip_scores(
        base_files: List[Path],
        intv_files: List[Path],
        prompts: List[str],
        max_samples: Optional[int] = None,
        device: str = "cuda",
) -> Dict:
    """Compute CLIP-T scores and paired comparison."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    n = min(len(base_files), len(intv_files), len(prompts))
    if max_samples:
        n = min(n, max_samples)

    base_scores = []
    intv_scores = []

    with torch.no_grad():
        for i in tqdm(range(n), desc="CLIP-T", leave=False):
            # Get prompt
            idx = int(base_files[i].stem)
            if idx < len(prompts):
                prompt = prompts[idx]
            else:
                prompt = prompts[i % len(prompts)]

            txt = tokenizer([prompt]).to(device)
            txt_feat = model.encode_text(txt)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            # Baseline
            with Image.open(base_files[i]) as img:
                img_b = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
            feat_b = model.encode_image(img_b)
            feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
            base_scores.append((feat_b @ txt_feat.T).item())

            # Intervention
            with Image.open(intv_files[i]) as img:
                img_i = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
            feat_i = model.encode_image(img_i)
            feat_i = feat_i / feat_i.norm(dim=-1, keepdim=True)
            intv_scores.append((feat_i @ txt_feat.T).item())

    base_scores = np.array(base_scores)
    intv_scores = np.array(intv_scores)
    delta = intv_scores - base_scores

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(intv_scores, base_scores)

    return {
        "clip_base_mean": float(base_scores.mean()),
        "clip_base_std": float(base_scores.std()),
        "clip_intv_mean": float(intv_scores.mean()),
        "clip_intv_std": float(intv_scores.std()),
        "clip_delta": float(delta.mean()),
        "clip_delta_std": float(delta.std()),
        "clip_p_value": float(p_value),
        "clip_n_samples": n,
    }


def compute_lpips_scores(
        base_files: List[Path],
        intv_files: List[Path],
        max_samples: Optional[int] = None,
        device: str = "cuda",
) -> Dict:
    """Compute paired LPIPS scores."""
    import lpips

    loss_fn = lpips.LPIPS(net='alex').to(device).eval()

    n = min(len(base_files), len(intv_files))
    if max_samples:
        n = min(n, max_samples)

    # Random sample if needed
    if max_samples and n > max_samples:
        random.seed(42)
        indices = random.sample(range(len(base_files)), max_samples)
        base_files = [base_files[i] for i in indices]
        intv_files = [intv_files[i] for i in indices]
        n = max_samples

    def to_tensor(p):
        with Image.open(p) as im:
            im = im.convert("RGB")
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        x = x[None].to(device)
        return 2 * x - 1  # [-1, 1]

    vals = []
    with torch.no_grad():
        for i in tqdm(range(n), desc="LPIPS", leave=False):
            x = to_tensor(base_files[i])
            y = to_tensor(intv_files[i])
            d = loss_fn(x, y).item()
            vals.append(d)

    vals = np.array(vals)

    return {
        "lpips_mean": float(vals.mean()),
        "lpips_std": float(vals.std()),
        "lpips_median": float(np.median(vals)),
        "lpips_p95": float(np.percentile(vals, 95)),
        "lpips_n_samples": n,
    }


def compute_fid_shift(
        base_dir: Path,
        intv_dir: Path,
        max_samples: Optional[int] = None,
        device: str = "cuda",
) -> Dict:
    """Compute FID shift between baseline and intervention."""
    from cleanfid.fid import get_folder_features, build_feature_extractor

    try:
        # Build model
        model = build_feature_extractor(mode="clean", device=torch.device(device))

        # Extract features
        print("    Extracting baseline features...")
        Fb = get_folder_features(str(base_dir), model=model, device=device, num_workers=4)

        print("    Extracting intervention features...")
        Fi = get_folder_features(str(intv_dir), model=model, device=device, num_workers=4)

        print(f"    Fb shape: {Fb.shape}, Fi shape: {Fi.shape}")

        # Check for issues
        if np.isnan(Fb).any() or np.isinf(Fb).any():
            print("    WARNING: Baseline features contain NaN/Inf!")
            return {"fid_shift": float("nan"), "fid_error": "Baseline has NaN/Inf"}

        if np.isnan(Fi).any() or np.isinf(Fi).any():
            print("    WARNING: Intervention features contain NaN/Inf!")
            return {"fid_shift": float("nan"), "fid_error": "Intervention has NaN/Inf"}

        # Ensure same number of samples
        n = min(Fb.shape[0], Fi.shape[0])
        if max_samples:
            n = min(n, max_samples)
        Fb = Fb[:n]
        Fi = Fi[:n]

        if n < 50:
            print(f"    WARNING: Only {n} samples, FID may be unreliable")

        # Compute FID with explicit numerical stability
        def fid_from_feats(X, Y, eps=1e-6):
            X = X.astype(np.float64)
            Y = Y.astype(np.float64)
            mu_x = X.mean(0)
            mu_y = Y.mean(0)
            sig_x = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
            sig_y = np.cov(Y, rowvar=False) + eps * np.eye(Y.shape[1])
            diff = mu_x - mu_y

            # Compute sqrt with better numerical stability
            covmean, _ = sqrtm(sig_x @ sig_y, disp=False)
            if np.iscomplexobj(covmean):
                if np.allclose(np.imag(covmean), 0, atol=1e-3):
                    covmean = np.real(covmean)
                else:
                    print(f"    WARNING: sqrtm has large imaginary component")
                    covmean = np.real(covmean)

            fid_val = float(diff @ diff + np.trace(sig_x + sig_y - 2 * covmean))

            # Sanity check
            if np.isnan(fid_val) or np.isinf(fid_val):
                print(f"    WARNING: FID is {fid_val}")
                return float("nan")

            return fid_val

        fid_shift = fid_from_feats(Fi, Fb)

        # Also compute paired L2 distance for sanity
        l2_dists = np.linalg.norm(Fb - Fi, axis=1)

        return {
            "fid_shift": float(fid_shift),
            "feature_l2_mean": float(l2_dists.mean()),
            "feature_l2_std": float(l2_dists.std()),
            "fid_n_samples": n,
        }

    except Exception as e:
        import traceback
        print(f"    ERROR in FID computation: {e}")
        traceback.print_exc()
        return {
            "fid_shift": float("nan"),
            "fid_error": str(e),
        }


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_experiment(name: str, config: Dict, device: str = "cuda") -> Dict:
    """Evaluate a single experiment."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {name}")
    print(f"Description: {config['description']}")
    print(f"{'=' * 60}")

    base_dir = Path(config["base_dir"])
    intv_dir = Path(config["intv_dir"])
    prompts_file = Path(config["prompts_file"])

    # Check directories exist
    if not base_dir.exists():
        print(f"  ERROR: Baseline directory not found: {base_dir}")
        return {"error": f"Baseline not found: {base_dir}"}

    if not intv_dir.exists():
        print(f"  ERROR: Intervention directory not found: {intv_dir}")
        return {"error": f"Intervention not found: {intv_dir}"}

    # Get aligned files
    base_files, intv_files = get_aligned_files(base_dir, intv_dir)
    n_images = len(base_files)
    print(f"  Found {n_images} aligned image pairs")

    if n_images == 0:
        return {"error": "No aligned images found"}

    # Load prompts
    if prompts_file.exists():
        prompts = load_prompts(prompts_file)
        print(f"  Loaded {len(prompts)} prompts")
    else:
        print(f"  WARNING: Prompts file not found, using dummy prompts")
        prompts = ["image"] * n_images

    results = {
        "name": name,
        "description": config["description"],
        "n_images": n_images,
    }

    # 1. CLIP-T
    print("\n  [1/3] Computing CLIP-T...")
    try:
        clip_results = compute_clip_scores(
            base_files, intv_files, prompts,
            max_samples=MAX_SAMPLES_CLIP,
            device=device,
        )
        results.update(clip_results)
        print(f"    CLIP-T: Δ = {clip_results['clip_delta']:+.4f}, p = {clip_results['clip_p_value']:.4f}")
    except Exception as e:
        print(f"    ERROR: {e}")
        results["clip_error"] = str(e)

    # 2. LPIPS
    print("\n  [2/3] Computing LPIPS...")
    try:
        lpips_results = compute_lpips_scores(
            base_files, intv_files,
            max_samples=MAX_SAMPLES_LPIPS,
            device=device,
        )
        results.update(lpips_results)
        print(f"    LPIPS: mean = {lpips_results['lpips_mean']:.4f} ± {lpips_results['lpips_std']:.4f}")
    except Exception as e:
        print(f"    ERROR: {e}")
        results["lpips_error"] = str(e)

    # 3. FID Shift
    print("\n  [3/3] Computing FID shift...")
    try:
        fid_results = compute_fid_shift(
            base_dir, intv_dir,
            max_samples=MAX_SAMPLES_FID,
            device=device,
        )
        results.update(fid_results)
        fid_val = fid_results.get('fid_shift', float('nan'))
        if np.isnan(fid_val):
            print(f"    FID shift: nan (see error above)")
        else:
            print(f"    FID shift: {fid_val:.2f}")
    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        traceback.print_exc()
        results["fid_error"] = str(e)
        results["fid_shift"] = float("nan")

    return results


def print_summary_table(all_results: Dict[str, Dict]):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Header
    print(f"{'Experiment':<20} {'N':>6} {'CLIP-T Δ':>10} {'p-value':>10} {'LPIPS':>10} {'FID shift':>12}")
    print("-" * 80)

    for name, results in all_results.items():
        if "error" in results and "n_images" not in results:
            print(f"{name:<20} ERROR: {results['error']}")
            continue

        n = results.get("n_images", "?")
        clip_delta = results.get("clip_delta", float("nan"))
        p_value = results.get("clip_p_value", float("nan"))
        lpips = results.get("lpips_mean", float("nan"))
        fid = results.get("fid_shift", float("nan"))

        # Format p-value
        if np.isnan(p_value):
            p_str = "nan"
        elif p_value < 0.001:
            p_str = "<0.001"
        elif p_value < 0.05:
            p_str = f"{p_value:.3f}*"
        else:
            p_str = f"{p_value:.3f}"

        # Format FID
        if np.isnan(fid):
            fid_str = "nan"
        else:
            fid_str = f"{fid:.1f}"

        # Format CLIP delta
        if np.isnan(clip_delta):
            clip_str = "nan"
        else:
            clip_str = f"{clip_delta:+.4f}"

        # Format LPIPS
        if np.isnan(lpips):
            lpips_str = "nan"
        else:
            lpips_str = f"{lpips:.4f}"

        print(f"{name:<20} {n:>6} {clip_str:>10} {p_str:>10} {lpips_str:>10} {fid_str:>12}")

    print("=" * 80)
    print("* p < 0.05 indicates significant difference")


def generate_latex_table(all_results: Dict[str, Dict], output_path: Path):
    """Generate LaTeX table."""
    latex = r"""\begin{table}[t]
\centering
\caption{\textbf{Quality metrics under attention sink removal.} 
$\Delta$CLIP-T measures change in text-image alignment.
LPIPS measures perceptual distance (lower = more similar to baseline).
FID measures distributional shift (lower = more similar).}
\label{tab:all_metrics}
\begin{tabular}{llccccc}
\toprule
Dataset & Condition & $N$ & $\Delta$CLIP-T & $p$ & LPIPS $\downarrow$ & FID $\downarrow$ \\
\midrule
"""

    # Group by dataset
    geneval_results = {k: v for k, v in all_results.items() if "GenEval" in k}
    coco_results = {k: v for k, v in all_results.items() if "COCO" in k}

    # GenEval rows
    first_geneval = True
    for name, results in geneval_results.items():
        if "error" in results and "n_images" not in results:
            continue

        n = results.get("n_images", "?")
        clip_delta = results.get("clip_delta", float("nan"))
        p_value = results.get("clip_p_value", float("nan"))
        lpips = results.get("lpips_mean", float("nan"))
        fid = results.get("fid_shift", float("nan"))

        # Extract condition name (A1, A2, A3)
        condition = name.split("_")[-1] if "_" in name else name

        # Format values
        if np.isnan(p_value):
            p_str = "--"
        elif p_value < 0.001:
            p_str = "$<$0.001"
        else:
            p_str = f"{p_value:.3f}"

        clip_str = f"${clip_delta:+.3f}$" if not np.isnan(clip_delta) else "--"
        lpips_str = f"{lpips:.3f}" if not np.isnan(lpips) else "--"
        fid_str = f"{fid:.0f}" if not np.isnan(fid) else "--"

        if first_geneval:
            latex += rf"\multirow{{{len(geneval_results)}}}{{*}}{{GenEval}}" + "\n"
            first_geneval = False

        latex += rf"  & {condition} & {n} & {clip_str} & {p_str} & {lpips_str} & {fid_str} \\" + "\n"

    latex += r"\midrule" + "\n"

    # COCO rows
    for name, results in coco_results.items():
        if "error" in results and "n_images" not in results:
            continue

        n = results.get("n_images", "?")
        clip_delta = results.get("clip_delta", float("nan"))
        p_value = results.get("clip_p_value", float("nan"))
        lpips = results.get("lpips_mean", float("nan"))
        fid = results.get("fid_shift", float("nan"))

        condition = "A1"

        if np.isnan(p_value):
            p_str = "--"
        elif p_value < 0.001:
            p_str = "$<$0.001"
        else:
            p_str = f"{p_value:.3f}"

        clip_str = f"${clip_delta:+.3f}$" if not np.isnan(clip_delta) else "--"
        lpips_str = f"{lpips:.3f}" if not np.isnan(lpips) else "--"
        fid_str = f"{fid:.0f}" if not np.isnan(fid) else "--"

        latex += rf"COCO-5k & {condition} & {n} & {clip_str} & {p_str} & {lpips_str} & {fid_str} \\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\vspace{1mm}

\small
\textit{A1: Layer 12, top-1 sink; A2: Multi-layer, top-1; A3: Layer 12, top-5.
Semantic alignment (CLIP-T) is preserved while perceptual change scales with intervention intensity.}
\end{table}
"""

    output_path.write_text(latex)
    print(f"\nLaTeX table saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate all experiments")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--output", default="all_results.json", help="Output JSON file")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Specific experiments to run (default: all)")
    args = parser.parse_args()

    # Select experiments
    if args.experiments:
        experiments = {k: v for k, v in EXPERIMENTS.items() if k in args.experiments}
    else:
        experiments = EXPERIMENTS

    print(f"Will evaluate {len(experiments)} experiments:")
    for name in experiments:
        print(f"  - {name}")

    # Run evaluations
    all_results = {}
    for name, config in experiments.items():
        results = evaluate_experiment(name, config, device=args.device)
        all_results[name] = results

    # Print summary
    print_summary_table(all_results)

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate LaTeX
    latex_path = output_path.with_suffix(".tex")
    generate_latex_table(all_results, latex_path)


if __name__ == "__main__":
    main()