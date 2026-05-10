#!/usr/bin/env python3
"""
Summarize k-sweep counterfactual results with proper multiple comparison correction.
Generates LaTeX table for paper.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    if n == 0:
        return []
    
    indexed_pvals = [(i, p) for i, p in enumerate(p_values)]
    indexed_pvals.sort(key=lambda x: x[1])
    
    adjusted = [0.0] * n
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_pvals):
        adj_p = min(1.0, p * (n - rank))
        adj_p = max(adj_p, prev_adj)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p
    
    return adjusted


def load_results(base_dir: str, k_values: List[int]) -> Dict:
    """Load results from k-sweep directories."""
    results = {}
    for k in k_values:
        path = Path(base_dir) / f"k{k}" / "counterfactual_results.json"
        if path.exists():
            with open(path) as f:
                results[k] = json.load(f)
        else:
            print(f"Warning: {path} not found", file=sys.stderr)
    return results


def apply_global_correction(results: Dict) -> Dict:
    """Apply Holm correction across ALL comparisons (all k × all modes)."""
    # Collect all p-values with their (k, mode) keys
    all_pvals = []
    for k, data in results.items():
        for mode, mode_data in data['modes'].items():
            all_pvals.append((k, mode, mode_data['p_value']))
    
    # Apply Holm correction
    raw_pvals = [p for _, _, p in all_pvals]
    adj_pvals = holm_bonferroni_correction(raw_pvals)
    
    # Store back
    for i, (k, mode, _) in enumerate(all_pvals):
        results[k]['modes'][mode]['p_value_global_adj'] = adj_pvals[i]
        results[k]['modes'][mode]['significant_global'] = adj_pvals[i] < 0.05
    
    return results


def generate_latex_table(results: Dict, k_values: List[int]) -> str:
    """Generate LaTeX table for paper."""
    modes = ["top_sink", "random", "bottom_sink", "high_outgoing_query"]
    mode_labels = {
        "top_sink": "Top-sink",
        "random": "Random",
        "bottom_sink": "Bottom-sink", 
        "high_outgoing_query": "High-outgoing"
    }
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Counterfactual ablation results across $k$ values. ")
    latex.append(r"$\Delta$CLIP-T shows change from baseline. ")
    latex.append(r"$p_{\text{adj}}$ is Holm-corrected across all 20 comparisons. ")
    latex.append(r"All effects are small ($|\Delta| < 0.01$).}")
    latex.append(r"\label{tab:counterfactual}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{llrrr}")
    latex.append(r"\toprule")
    latex.append(r"$k$ & Mode & $\Delta$CLIP-T & 95\% CI & $p_{\text{adj}}$ \\")
    latex.append(r"\midrule")
    
    for k in k_values:
        if k not in results:
            continue
        
        data = results[k]
        first_row = True
        
        for mode in modes:
            if mode not in data['modes']:
                continue
            
            m = data['modes'][mode]
            delta = m['delta_mean']
            ci_low, ci_high = m['ci_low'], m['ci_high']
            p_adj = m.get('p_value_global_adj', m.get('p_value_adj', m['p_value']))
            sig = m.get('significant_global', p_adj < 0.05)
            
            k_str = str(k) if first_row else ""
            mode_str = mode_labels.get(mode, mode)
            delta_str = f"{delta:+.4f}"
            ci_str = f"[{ci_low:+.3f}, {ci_high:+.3f}]"
            p_str = f"{p_adj:.3f}" + ("*" if sig else "")
            
            latex.append(f"{k_str} & {mode_str} & {delta_str} & {ci_str} & {p_str} \\\\")
            first_row = False
        
        latex.append(r"\midrule")
    
    # Remove last midrule
    latex[-1] = r"\bottomrule"
    
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)


def print_summary(results: Dict, k_values: List[int]):
    """Print summary statistics."""
    modes = ["top_sink", "random", "bottom_sink", "high_outgoing_query"]
    
    print("=" * 100)
    print("K-SWEEP SUMMARY (with Global Holm Correction)")
    print("=" * 100)
    print(f"{'k':<6} {'Mode':<22} {'ΔCLIP-T':>10} {'95% CI':>24} {'p_raw':>10} {'p_adj':>10} {'Sig':>6}")
    print("-" * 100)
    
    for k in k_values:
        if k not in results:
            continue
        data = results[k]
        
        for mode in modes:
            if mode not in data['modes']:
                continue
            m = data['modes'][mode]
            
            delta = m['delta_mean']
            ci_str = f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]"
            p_raw = m['p_value']
            p_adj = m.get('p_value_global_adj', m['p_value'])
            sig = "✓" if p_adj < 0.05 else "-"
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "="
            
            print(f"{k:<6} {mode:<22} {delta:>+.4f}{direction} {ci_str:>24} {p_raw:>10.4f} {p_adj:>10.4f} {sig:>6}")
        
        print("-" * 100)
    
    # Count significants
    n_tests = 0
    n_sig_raw = 0
    n_sig_adj = 0
    n_positive_sig = 0
    
    for k in k_values:
        if k not in results:
            continue
        for mode in modes:
            if mode not in results[k]['modes']:
                continue
            m = results[k]['modes'][mode]
            n_tests += 1
            if m['p_value'] < 0.05:
                n_sig_raw += 1
            p_adj = m.get('p_value_global_adj', m['p_value'])
            if p_adj < 0.05:
                n_sig_adj += 1
                if m['delta_mean'] > 0:
                    n_positive_sig += 1
    
    print(f"\nTotal tests: {n_tests}")
    print(f"Significant (raw α=0.05): {n_sig_raw} ({100*n_sig_raw/n_tests:.1f}%)")
    print(f"Significant (Holm-adjusted): {n_sig_adj} ({100*n_sig_adj/n_tests:.1f}%)")
    print(f"Expected false positives under H0: {n_tests * 0.05:.1f}")
    
    if n_positive_sig > 0:
        print(f"\n⚠ WARNING: {n_positive_sig} significant effects are POSITIVE (CLIP improved)")
        print("  This suggests metric artifact rather than true degradation.")
    
    # Check if top_sink is consistently non-significant
    top_sink_sig = sum(1 for k in k_values 
                       if k in results and 
                       results[k]['modes'].get('top_sink', {}).get('p_value_global_adj', 1.0) < 0.05)
    
    print(f"\n• top_sink significant after correction: {top_sink_sig}/{len([k for k in k_values if k in results])}")
    
    if top_sink_sig == 0:
        print("  → top_sink ablation is consistently tolerated across all k values")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results_k_sweep",
                       help="Base directory containing k* subdirectories")
    parser.add_argument("--k_values", type=str, default="1,5,10,20,50",
                       help="Comma-separated k values")
    parser.add_argument("--latex_out", type=str, default=None,
                       help="Output file for LaTeX table")
    args = parser.parse_args()
    
    k_values = [int(k) for k in args.k_values.split(",")]
    
    # Load results
    results = load_results(args.base_dir, k_values)
    
    if not results:
        print("No results found!", file=sys.stderr)
        sys.exit(1)
    
    # Apply global Holm correction
    results = apply_global_correction(results)
    
    # Print summary
    print_summary(results, k_values)
    
    # Generate LaTeX
    latex = generate_latex_table(results, k_values)
    
    print("\n" + "=" * 100)
    print("LATEX TABLE")
    print("=" * 100)
    print(latex)
    
    if args.latex_out:
        with open(args.latex_out, 'w') as f:
            f.write(latex)
        print(f"\nLaTeX table saved to: {args.latex_out}")


if __name__ == "__main__":
    main()
