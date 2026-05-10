#!/usr/bin/env python
"""
HPS-v2 Evaluator Module (Production Version v3 - Final)

修复内容:
1. 错误返回 NaN（不是 0.0），自动剔除并报告
2. 不自动安装依赖
3. API 兼容：支持 list/float/ndarray/torch.Tensor
4. 文件句柄：使用 with 正确关闭
5. 错误率 > 2% 时 abort
6. PIL.Image 分支也 .copy()

使用方式:
    from hpsv2_evaluator import HPSv2Evaluator, evaluate_experiment_hpsv2
    
    stats, baseline_scores, intervention_scores = evaluate_experiment_hpsv2(
        baseline_images, intervention_images, prompts, output_dir
    )
"""

import numpy as np
from PIL import Image
from typing import List, Union, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import json
from scipy import stats
import warnings


def import_hpsv2():
    """
    导入 hpsv2，不自动安装。
    如果未安装则报错并提示用户安装。
    """
    try:
        import hpsv2
        return hpsv2
    except ImportError as e:
        raise ImportError(
            "hpsv2 is not installed. Please install it manually:\n"
            "  pip install hpsv2\n"
            "Then re-run this script."
        ) from e


def safe_hps_score(hpsv2_module, img: Image.Image, prompt: str, version: str = "v2.1") -> float:
    """
    安全地计算 HPS-v2 分数，兼容多种返回格式。
    
    支持的返回类型:
    - list/tuple: [score]
    - np.ndarray: array([score])
    - torch.Tensor: tensor([score]) 或 tensor(score)
    - float/int: score
    
    Args:
        hpsv2_module: hpsv2 模块
        img: PIL Image (RGB)
        prompt: 文本提示
        version: HPS 版本 ("v2.0" 或 "v2.1")
        
    Returns:
        float: HPS-v2 分数
        
    Raises:
        RuntimeError: 如果计算失败
    """
    try:
        out = hpsv2_module.score(img, prompt, hps_version=version)
        
        # list/tuple
        if isinstance(out, (list, tuple)):
            return float(out[0])
        
        # numpy ndarray
        if isinstance(out, np.ndarray):
            return float(out.reshape(-1)[0])
        
        # torch.Tensor
        try:
            import torch
            if isinstance(out, torch.Tensor):
                return float(out.detach().cpu().reshape(-1)[0].item())
        except ImportError:
            pass
        except Exception:
            pass
        
        # python scalar (float/int)
        return float(out)
        
    except Exception as e:
        raise RuntimeError(f"HPSv2 scoring failed: {e}") from e


class HPSv2Evaluator:
    """
    HPS-v2 评估器（生产版本）
    
    关键特性:
    - 错误返回 NaN（不是 0.0），防止伪阴性
    - 文件句柄正确关闭
    - 错误率检查
    - 支持 torch.Tensor 返回格式
    """
    
    def __init__(self, version: str = "v2.1", strict: bool = False):
        """
        初始化 HPS-v2 评估器
        
        Args:
            version: HPS 版本 ("v2.0" 或 "v2.1")
            strict: 如果为 True，任何错误立即抛出；否则返回 NaN
        """
        self.version = version
        self.strict = strict
        self.hpsv2 = import_hpsv2()
        self._error_count = 0
        self._total_count = 0
    
    def score(self, image: Union[Image.Image, str, Path], prompt: str) -> float:
        """
        计算单个图像的 HPS-v2 分数
        
        Args:
            image: PIL Image 或图像路径
            prompt: 对应的文本提示
            
        Returns:
            HPS-v2 分数；失败时返回 np.nan（除非 strict=True）
        """
        self._total_count += 1
        
        try:
            # 加载图像，使用 with 确保句柄关闭
            if isinstance(image, (str, Path)):
                with Image.open(image) as im:
                    img = im.convert('RGB').copy()
            else:
                # PIL.Image 分支也 copy()，避免潜在共享缓冲问题
                img = image.convert('RGB').copy() if image.mode != 'RGB' else image.copy()
            
            return safe_hps_score(self.hpsv2, img, prompt, self.version)
            
        except Exception as e:
            self._error_count += 1
            if self.strict:
                raise RuntimeError(
                    f"HPSv2 scoring failed for prompt '{prompt[:50]}...': {e}"
                ) from e
            else:
                return np.nan
    
    def score_batch(
        self, 
        images: List[Union[Image.Image, str, Path]], 
        prompts: List[str],
        show_progress: bool = True,
        max_error_rate: float = 0.02,
    ) -> Tuple[np.ndarray, int]:
        """
        批量计算 HPS-v2 分数
        
        Args:
            images: 图像列表
            prompts: 对应的提示列表
            show_progress: 是否显示进度条
            max_error_rate: 允许的最大错误率，超过则抛出异常
            
        Returns:
            (scores, n_errors): 分数数组（可能包含 NaN）和错误数量
            
        Raises:
            RuntimeError: 如果错误率超过 max_error_rate
        """
        self._error_count = 0
        self._total_count = 0
        
        scores = []
        iterator = list(zip(images, prompts))
        if show_progress:
            iterator = tqdm(iterator, desc="HPSv2")
        
        for image, prompt in iterator:
            score = self.score(image, prompt)
            scores.append(score)
        
        scores = np.array(scores, dtype=np.float64)
        n_errors = self._error_count
        
        if n_errors > 0:
            error_rate = n_errors / len(scores)
            print(f"  HPSv2 errors: {n_errors}/{len(scores)} ({error_rate:.1%})")
            
            if error_rate > max_error_rate:
                raise RuntimeError(
                    f"HPSv2 error rate ({error_rate:.1%}) exceeds threshold ({max_error_rate:.1%}). "
                    f"Aborting to prevent biased results."
                )
        
        return scores, n_errors


def compute_paired_stats(
    baseline_scores: np.ndarray,
    intervention_scores: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    min_valid_ratio: float = 0.98,
) -> dict:
    """
    计算配对统计量，自动剔除 NaN 样本
    """
    mask = np.isfinite(baseline_scores) & np.isfinite(intervention_scores)
    n_original = len(baseline_scores)
    n_valid = int(np.sum(mask))
    n_dropped = n_original - n_valid
    
    if n_dropped > 0:
        print(f"  Dropped {n_dropped}/{n_original} samples with NaN scores")
    
    valid_ratio = n_valid / n_original if n_original > 0 else 0
    if valid_ratio < min_valid_ratio:
        raise RuntimeError(
            f"Too few valid samples: {n_valid}/{n_original} ({valid_ratio:.1%} < {min_valid_ratio:.0%}). "
            f"Aborting."
        )
    
    baseline = baseline_scores[mask]
    intervention = intervention_scores[mask]
    
    deltas = intervention - baseline
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)
    
    rng = np.random.RandomState(seed)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(deltas), len(deltas), replace=True)
        bootstrap_means.append(np.mean(deltas[idx]))
    
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    t_stat, p_value = stats.ttest_rel(intervention, baseline)
    
    ci_includes_zero = ci_lower <= 0 <= ci_upper
    
    return {
        'baseline_mean': float(np.mean(baseline)),
        'baseline_std': float(np.std(baseline, ddof=1)),
        'intervention_mean': float(np.mean(intervention)),
        'intervention_std': float(np.std(intervention, ddof=1)),
        'delta_mean': float(delta_mean),
        'delta_std': float(delta_std),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'ci_includes_zero': ci_includes_zero,
        'n_samples': n_valid,
        'n_dropped': n_dropped,
    }


def evaluate_experiment_hpsv2(
    baseline_images: List[Union[Image.Image, Path, str]],
    intervention_images: List[Union[Image.Image, Path, str]],
    prompts: List[str],
    output_dir: Optional[Path] = None,
    version: str = "v2.1",
    strict: bool = False,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    对实验结果进行 HPS-v2 评估（主入口函数）
    
    注意：调用者必须确保 baseline_images, intervention_images, prompts 
    已严格对齐（同一索引对应同一 prompt）。
    """
    if len(baseline_images) != len(intervention_images):
        raise ValueError(
            f"baseline ({len(baseline_images)}) and intervention ({len(intervention_images)}) "
            f"image counts don't match."
        )
    if len(baseline_images) != len(prompts):
        raise ValueError(
            f"Image count ({len(baseline_images)}) and prompt count ({len(prompts)}) don't match."
        )
    
    evaluator = HPSv2Evaluator(version=version, strict=strict)
    
    print("\n" + "="*60)
    print("HPS-v2 EVALUATION")
    print("="*60)
    print(f"Samples: {len(prompts)}")
    print(f"Version: {version}")
    
    print("\nScoring baseline images...")
    baseline_scores, baseline_errors = evaluator.score_batch(baseline_images, prompts)
    
    print("\nScoring intervention images...")
    intervention_scores, intervention_errors = evaluator.score_batch(intervention_images, prompts)
    
    print("\nComputing paired statistics...")
    stats_dict = compute_paired_stats(baseline_scores, intervention_scores)
    stats_dict['baseline_errors'] = baseline_errors
    stats_dict['intervention_errors'] = intervention_errors
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Valid samples: {stats_dict['n_samples']} (dropped: {stats_dict['n_dropped']})")
    print(f"Baseline:      {stats_dict['baseline_mean']:.4f} ± {stats_dict['baseline_std']:.4f}")
    print(f"Intervention:  {stats_dict['intervention_mean']:.4f} ± {stats_dict['intervention_std']:.4f}")
    print(f"Δ:             {stats_dict['delta_mean']:+.4f}")
    print(f"95% CI:        [{stats_dict['ci_lower']:+.4f}, {stats_dict['ci_upper']:+.4f}]")
    print(f"p-value:       {stats_dict['p_value']:.4f}")
    
    if stats_dict['ci_includes_zero']:
        print(f"\n✓ CI includes 0: no significant quality degradation")
    else:
        if stats_dict['delta_mean'] > 0:
            print(f"\n→ CI > 0: slight improvement")
        else:
            print(f"\n⚠ CI < 0: potential degradation")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'hpsv2_baseline_scores.npy', baseline_scores)
        np.save(output_dir / 'hpsv2_intervention_scores.npy', intervention_scores)
        
        with open(output_dir / 'hpsv2_stats.json', 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
    
    return stats_dict, baseline_scores, intervention_scores


if __name__ == '__main__':
    print("HPSv2 Evaluator Module (Production Version v3 - Final)")
