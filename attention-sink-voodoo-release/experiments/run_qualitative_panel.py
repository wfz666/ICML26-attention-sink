#!/usr/bin/env python3
"""
Rebuttal P2: Qualitative Drift Panel & Taxonomy
=================================================

Reads P1 results (images + per-sample scores) and produces:

1. A ranked selection of representative paired examples
2. Side-by-side comparison panels (baseline | sink | random)
3. A combined rebuttal figure with annotations
4. A CSV template for manual taxonomy annotation
5. Auto-detected change categories via simple image analysis

Addresses all four reviewers' concerns about the *nature* of
perceptual drift. The goal is to show: most shifts are
texture/style/lighting, not compositional failures.

Usage:
    python rebuttal_qualitative_panel.py \\
        --p1_dir results_rebuttal_p1 \\
        --output_dir results_rebuttal_p2 \\
        --k 1

    # Also generate k=5 panels
    python rebuttal_qualitative_panel.py \\
        --p1_dir results_rebuttal_p1 \\
        --output_dir results_rebuttal_p2 \\
        --k 1,5
"""

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional: matplotlib for combined figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================
# Load P1 data
# ============================================================

def load_p1_data(p1_dir: Path, k: int) -> Dict:
    """Load per-sample scores and prompts from P1 results."""
    results_file = p1_dir / f"results_k{k}.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Missing {results_file}. Run rebuttal_perceptual_delta_delta.py first.")

    with open(results_file) as f:
        results = json.load(f)

    with open(p1_dir / "prompts.txt") as f:
        prompts = [line.strip() for line in f if line.strip()]

    raw = results["raw_scores"]
    n = len(prompts)

    data = {
        "prompts": prompts,
        "n": n,
        "k": k,
        "clip_base": np.array(raw["clip_base"]),
        "clip_sink": np.array(raw["clip_sink"]),
        "clip_rand": np.array(raw["clip_rand"]),
        "lpips_sink": np.array(raw["lpips_sink"]),
        "lpips_rand": np.array(raw["lpips_rand"]),
    }

    # Derived
    data["delta_clip_sink"] = data["clip_sink"] - data["clip_base"]
    data["delta_clip_rand"] = data["clip_rand"] - data["clip_base"]

    return data


def load_triplet(p1_dir: Path, k: int, idx: int) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Load (baseline, sink, random) image triplet."""
    base = Image.open(p1_dir / f"images_k{k}_none" / f"{idx:04d}.png").convert("RGB")
    sink = Image.open(p1_dir / f"images_k{k}_top_sink" / f"{idx:04d}.png").convert("RGB")
    rand = Image.open(p1_dir / f"images_k{k}_random" / f"{idx:04d}.png").convert("RGB")
    return base, sink, rand


# ============================================================
# Sample selection
# ============================================================

def select_samples(data: Dict, n_per_group: int = 5) -> Dict[str, List[int]]:
    """
    Select representative samples for the qualitative panel.

    Groups:
      A. High LPIPS_sink + CLIP-T stable  (typical drift)
      B. Medium LPIPS_sink + CLIP-T stable (mild drift)
      C. Largest |ΔCLIP-T| under sink masking (boundary cases)
      D. Low LPIPS_sink (near-identity, sinks had minimal effect)
    """
    n = data["n"]
    lpips_sink = data["lpips_sink"]
    delta_clip = data["delta_clip_sink"]

    # Sort indices
    rank_lpips_desc = np.argsort(-lpips_sink)  # highest LPIPS first
    rank_lpips_asc = np.argsort(lpips_sink)    # lowest LPIPS first
    rank_clip_change = np.argsort(-np.abs(delta_clip))  # largest |ΔCLIP| first

    # Stable CLIP-T: |ΔCLIP| < 0.01
    clip_stable = np.abs(delta_clip) < 0.01

    used = set()

    def pick(candidates, count):
        picked = []
        for idx in candidates:
            if idx not in used and len(picked) < count:
                picked.append(int(idx))
                used.add(idx)
        return picked

    groups = {}

    # A: High LPIPS, CLIP stable
    candidates_a = [i for i in rank_lpips_desc if clip_stable[i]]
    groups["A_high_lpips_stable_clip"] = pick(candidates_a, n_per_group)

    # B: Medium LPIPS (25th-75th percentile), CLIP stable
    p25, p75 = np.percentile(lpips_sink, 25), np.percentile(lpips_sink, 75)
    candidates_b = [i for i in range(n) if p25 <= lpips_sink[i] <= p75 and clip_stable[i]]
    # Sort by LPIPS for cleaner ordering
    candidates_b.sort(key=lambda i: lpips_sink[i])
    # Pick from the middle
    mid = len(candidates_b) // 2
    start = max(0, mid - n_per_group)
    candidates_b = candidates_b[start:]
    groups["B_medium_lpips_stable_clip"] = pick(candidates_b, n_per_group)

    # C: Largest CLIP-T change (might reveal compositional issues)
    groups["C_largest_clip_change"] = pick(rank_clip_change, n_per_group)

    # D: Lowest LPIPS (sinks barely mattered)
    candidates_d = [i for i in rank_lpips_asc]
    groups["D_low_lpips_minimal_shift"] = pick(candidates_d, n_per_group)

    return groups


# ============================================================
# Simple automated change analysis
# ============================================================

def analyze_change(base: Image.Image, modified: Image.Image) -> Dict:
    """
    Simple pixel-level analysis of what changed.
    Returns dict with change statistics.
    """
    a = np.array(base).astype(np.float32)
    b = np.array(modified).astype(np.float32)

    diff = np.abs(a - b)
    diff_gray = diff.mean(axis=-1)  # average over RGB

    # Overall change magnitude
    mean_diff = float(diff_gray.mean())
    max_diff = float(diff_gray.max())

    # Spatial distribution: what fraction of pixels changed significantly?
    threshold = 20.0  # out of 255
    changed_mask = diff_gray > threshold
    frac_changed = float(changed_mask.mean())

    # Where are changes concentrated? Divide into quadrants
    h, w = diff_gray.shape
    quadrants = {
        "top_left": diff_gray[:h//2, :w//2].mean(),
        "top_right": diff_gray[:h//2, w//2:].mean(),
        "bottom_left": diff_gray[h//2:, :w//2].mean(),
        "bottom_right": diff_gray[h//2:, w//2:].mean(),
    }
    max_q = max(quadrants, key=quadrants.get)
    min_q = min(quadrants, key=quadrants.get)
    spatial_uniformity = min(quadrants.values()) / (max(quadrants.values()) + 1e-8)

    # Color channel analysis: is change in one channel dominant?
    channel_diffs = [float(diff[:, :, c].mean()) for c in range(3)]
    dominant_channel = ["R", "G", "B"][np.argmax(channel_diffs)]
    channel_uniformity = min(channel_diffs) / (max(channel_diffs) + 1e-8)

    # High-frequency vs low-frequency (rough proxy)
    # Downsample both, compute diff on downsampled -> low-freq component
    base_small = np.array(base.resize((64, 64), Image.BILINEAR)).astype(np.float32)
    mod_small = np.array(modified.resize((64, 64), Image.BILINEAR)).astype(np.float32)
    low_freq_diff = float(np.abs(base_small - mod_small).mean())
    high_freq_ratio = mean_diff / (low_freq_diff + 1e-8)

    # Heuristic change type
    if frac_changed < 0.05:
        change_type = "minimal"
    elif spatial_uniformity > 0.7 and channel_uniformity > 0.7:
        change_type = "global_style"  # uniform change across space and channels
    elif spatial_uniformity < 0.4:
        change_type = "local_structural"  # concentrated in specific regions
    elif high_freq_ratio > 1.5:
        change_type = "texture_detail"  # mostly high-frequency changes
    else:
        change_type = "mixed"

    return {
        "mean_pixel_diff": round(mean_diff, 2),
        "max_pixel_diff": round(max_diff, 2),
        "frac_pixels_changed": round(frac_changed, 4),
        "spatial_uniformity": round(spatial_uniformity, 3),
        "channel_uniformity": round(channel_uniformity, 3),
        "dominant_channel": dominant_channel,
        "high_freq_ratio": round(high_freq_ratio, 3),
        "auto_category": change_type,
        "concentrated_quadrant": max_q,
    }


# ============================================================
# Panel generation (PIL-based, no matplotlib dependency)
# ============================================================

def make_triplet_panel(
    base: Image.Image,
    sink: Image.Image,
    rand: Image.Image,
    prompt: str,
    metrics: Dict,
    panel_width: int = 1800,
) -> Image.Image:
    """Create a single triplet comparison panel with annotations."""

    # Resize all to same size
    img_w = panel_width // 3 - 10
    img_h = int(img_w * base.height / base.width)

    base_r = base.resize((img_w, img_h), Image.LANCZOS)
    sink_r = sink.resize((img_w, img_h), Image.LANCZOS)
    rand_r = rand.resize((img_w, img_h), Image.LANCZOS)

    # Create canvas
    header_h = 70
    footer_h = 50
    total_h = header_h + img_h + footer_h
    canvas = Image.new("RGB", (panel_width, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Try to load a font; fall back to default
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_metric = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_small = font_title
        font_metric = font_title

    # Header: prompt (truncated)
    prompt_display = prompt[:120] + "..." if len(prompt) > 120 else prompt
    draw.text((10, 5), f'"{prompt_display}"', fill="black", font=font_small)

    # Paste images
    x_positions = [5, img_w + 10, 2 * img_w + 15]
    labels = ["Baseline", "Sink-removed", "Random-removed"]
    y_img = header_h

    for x, img, label in zip(x_positions, [base_r, sink_r, rand_r], labels):
        canvas.paste(img, (x, y_img))
        draw.text((x + 5, y_img - 18), label, fill="black", font=font_title)

    # Footer: metrics
    lpips_s = metrics.get("lpips_sink", 0)
    lpips_r = metrics.get("lpips_rand", 0)
    dclip_s = metrics.get("delta_clip_sink", 0)
    auto_cat = metrics.get("auto_category", "?")

    footer_y = y_img + img_h + 5
    draw.text(
        (10, footer_y),
        f"LPIPS: sink={lpips_s:.3f}  rand={lpips_r:.3f}  |  "
        f"ΔCLIP-T: sink={dclip_s:+.4f}  |  "
        f"Auto: {auto_cat}",
        fill="gray", font=font_metric,
    )

    return canvas


# ============================================================
# Combined rebuttal figure (matplotlib)
# ============================================================

def make_rebuttal_figure(
    panels: List[Dict],
    output_path: Path,
    k: int,
    max_rows: int = 5,
):
    """Create a compact multi-row figure for the rebuttal PDF."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping combined figure")
        return

    # Select top rows by LPIPS (most informative)
    panels_sorted = sorted(panels, key=lambda p: -p["lpips_sink"])
    selected = panels_sorted[:max_rows]

    fig, axes = plt.subplots(max_rows, 3, figsize=(14, 3.2 * max_rows))
    if max_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Baseline", "Sink-removed", "Random-removed"]

    for row, panel in enumerate(selected):
        for col, key in enumerate(["base", "sink", "rand"]):
            ax = axes[row, col]
            ax.imshow(np.array(panel[key]))
            ax.axis("off")

            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight="bold")

        # Row annotation on the right
        prompt_short = panel["prompt"][:50] + "..." if len(panel["prompt"]) > 50 else panel["prompt"]
        axes[row, 2].text(
            1.02, 0.5,
            f'LPIPS={panel["lpips_sink"]:.3f}\n'
            f'ΔCLIP={panel["delta_clip_sink"]:+.4f}\n'
            f'{panel["auto_category"]}',
            transform=axes[row, 2].transAxes,
            fontsize=9, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
        )

    plt.suptitle(
        f"Qualitative Drift Panel (k={k}): Baseline vs Sink-removed vs Random-removed",
        fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 0.98])
    fig_path = output_path / f"rebuttal_drift_panel_k{k}.pdf"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved combined figure: {fig_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Rebuttal P2: Qualitative drift panel")
    parser.add_argument("--p1_dir", type=str, default="results_rebuttal_p1",
                        help="Directory from rebuttal_perceptual_delta_delta.py")
    parser.add_argument("--output_dir", type=str, default="results_rebuttal_p2")
    parser.add_argument("--k", type=str, default="1",
                        help="Comma-separated k values (e.g., '1' or '1,5')")
    parser.add_argument("--n_per_group", type=int, default=5)
    args = parser.parse_args()

    p1_dir = Path(args.p1_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    k_values = [int(x) for x in args.k.split(",")]

    for k in k_values:
        print(f"\n{'=' * 70}")
        print(f"  Qualitative Panel: k={k}")
        print(f"{'=' * 70}")

        # Load data
        data = load_p1_data(p1_dir, k)
        print(f"  Loaded {data['n']} samples")

        # Select representative samples
        groups = select_samples(data, n_per_group=args.n_per_group)
        total_selected = sum(len(v) for v in groups.values())
        print(f"  Selected {total_selected} samples across {len(groups)} groups:")
        for group_name, indices in groups.items():
            print(f"    {group_name}: {indices}")

        # Process each sample
        all_panels = []
        all_rows_for_csv = []

        for group_name, indices in groups.items():
            group_dir = output_path / f"k{k}_{group_name}"
            group_dir.mkdir(parents=True, exist_ok=True)

            for idx in indices:
                base, sink, rand = load_triplet(p1_dir, k, idx)

                # Automated change analysis
                analysis_sink = analyze_change(base, sink)
                analysis_rand = analyze_change(base, rand)

                metrics = {
                    "lpips_sink": data["lpips_sink"][idx],
                    "lpips_rand": data["lpips_rand"][idx],
                    "delta_clip_sink": data["delta_clip_sink"][idx],
                    "delta_clip_rand": data["delta_clip_rand"][idx],
                    "auto_category": analysis_sink["auto_category"],
                }

                # Generate triplet panel
                panel_img = make_triplet_panel(
                    base, sink, rand, data["prompts"][idx], metrics,
                )
                panel_img.save(group_dir / f"{idx:04d}_triplet.png")

                # Also save individual images for convenience
                base.save(group_dir / f"{idx:04d}_baseline.png")
                sink.save(group_dir / f"{idx:04d}_sink.png")
                rand.save(group_dir / f"{idx:04d}_random.png")

                # Collect for combined figure
                all_panels.append({
                    "idx": idx,
                    "group": group_name,
                    "prompt": data["prompts"][idx],
                    "base": base,
                    "sink": sink,
                    "rand": rand,
                    **metrics,
                    "sink_analysis": analysis_sink,
                    "rand_analysis": analysis_rand,
                })

                # CSV row
                all_rows_for_csv.append({
                    "idx": idx,
                    "group": group_name,
                    "prompt": data["prompts"][idx],
                    "lpips_sink": round(data["lpips_sink"][idx], 4),
                    "lpips_rand": round(data["lpips_rand"][idx], 4),
                    "delta_clip_sink": round(float(data["delta_clip_sink"][idx]), 4),
                    "delta_clip_rand": round(float(data["delta_clip_rand"][idx]), 4),
                    "auto_category_sink": analysis_sink["auto_category"],
                    "auto_category_rand": analysis_rand["auto_category"],
                    "frac_changed_sink": analysis_sink["frac_pixels_changed"],
                    "spatial_uniformity_sink": analysis_sink["spatial_uniformity"],
                    "high_freq_ratio_sink": analysis_sink["high_freq_ratio"],
                    # Manual annotation columns (fill in by hand)
                    "manual_category": "",
                    "compositional_error": "",
                    "notes": "",
                })

        # Save CSV for manual annotation
        csv_path = output_path / f"taxonomy_k{k}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows_for_csv[0].keys())
            writer.writeheader()
            writer.writerows(all_rows_for_csv)
        print(f"\n  Taxonomy CSV: {csv_path}")
        print(f"  → Fill in 'manual_category' and 'compositional_error' columns by hand")

        # Auto-category summary
        auto_cats = [r["auto_category_sink"] for r in all_rows_for_csv]
        cat_counts = {}
        for c in auto_cats:
            cat_counts[c] = cat_counts.get(c, 0) + 1
        print(f"\n  Auto-detected change types (sink masking):")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}/{len(auto_cats)}")

        # Combined rebuttal figure
        make_rebuttal_figure(all_panels, output_path, k, max_rows=5)

        # Save analysis JSON
        analysis_json = []
        for panel in all_panels:
            analysis_json.append({
                "idx": panel["idx"],
                "group": panel["group"],
                "prompt": panel["prompt"],
                "lpips_sink": float(panel["lpips_sink"]),
                "lpips_rand": float(panel["lpips_rand"]),
                "delta_clip_sink": float(panel["delta_clip_sink"]),
                "auto_category": panel["auto_category"],
                "sink_analysis": {k: float(v) if isinstance(v, (np.floating,)) else v
                                  for k, v in panel["sink_analysis"].items()},
                "rand_analysis": {k: float(v) if isinstance(v, (np.floating,)) else v
                                  for k, v in panel["rand_analysis"].items()},
            })
        with open(output_path / f"analysis_k{k}.json", "w") as f:
            json.dump(analysis_json, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    # ===== Cross-k summary =====
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Output directory: {output_path}")
    print(f"\nFiles per k-value:")
    print(f"  taxonomy_k*.csv         ← fill in manual_category + compositional_error")
    print(f"  analysis_k*.json        ← automated analysis details")
    print(f"  rebuttal_drift_panel_k*.pdf  ← 5-row comparison figure for rebuttal")
    print(f"  k*_A_high_lpips_*/      ← individual triplet panels")
    print(f"  k*_B_medium_lpips_*/")
    print(f"  k*_C_largest_clip_*/")
    print(f"  k*_D_low_lpips_*/")
    print(f"\nWorkflow:")
    print(f"  1. Look at the PDF panels to get an overview")
    print(f"  2. Open taxonomy_k*.csv, look at the triplet images for each row")
    print(f"  3. Fill in 'manual_category': texture, lighting, layout, color, composition, identity, minimal")
    print(f"  4. Fill in 'compositional_error': yes/no (object missing, wrong count, attribute swap)")
    print(f"  5. Use the filled CSV to report category frequencies in the rebuttal")


if __name__ == "__main__":
    main()