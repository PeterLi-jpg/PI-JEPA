#!/usr/bin/env python
"""
End-to-End Self-Supervised Pipeline for PI-JEPA.

This script orchestrates the complete self-supervised learning workflow:
1. Self-supervised pretraining on unlabeled coefficient fields
2. Supervised finetuning with frozen encoder on labeled data
3. Data efficiency comparison against baselines
4. Visualization of data efficiency curves

Validates: All Requirements (Req 1-9)
- Req 1: Self-supervised pretraining on unlabeled data
- Req 2: Physics-informed pretraining regularization
- Req 3: Spatial block masking strategy
- Req 4: Finetuning with frozen encoder
- Req 5: Full finetuning option
- Req 6: Data efficiency comparison framework
- Req 7: Unlabeled data loading
- Req 8: Checkpoint compatibility
- Req 9: Configuration updates

Usage:
    # Run full pipeline
    python scripts/run_self_supervised_pipeline.py --config configs/darcy.yaml
    
    # Skip pretraining if checkpoint exists
    python scripts/run_self_supervised_pipeline.py --skip-pretrain \
        --pretrain-checkpoint outputs/pretrain/checkpoint_best.pt
    
    # Run with visualization
    python scripts/run_self_supervised_pipeline.py --visualize
    
    # Custom labeled sample sweep
    python scripts/run_self_supervised_pipeline.py --n-labeled 10 25 50 100
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime

import torch

# Add PI-JEPA directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_data_efficiency_curves(
    results: Dict[str, Dict[int, float]],
    output_path: str = "outputs/data_efficiency/data_efficiency_curves.png",
    title: str = "Data Efficiency Comparison: PI-JEPA vs Baselines",
    log_scale: bool = True
) -> str:
    """
    Generate data efficiency curves comparing PI-JEPA against baselines.
    
    Creates a plot showing relative L2 error vs number of labeled samples
    for PI-JEPA and baseline models.
    
    Args:
        results: Dict mapping model name to {n_labeled: relative_l2_error}
        output_path: Path to save the figure
        title: Plot title
        log_scale: Whether to use log scale for x-axis
        
    Returns:
        Path to saved figure
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return ""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color and marker schemes
    colors = {
        'pi_jepa': '#2E86AB',      # Blue
        'fno': '#A23B72',          # Magenta
        'geo_fno': '#F18F01',      # Orange
        'deeponet': '#C73E1D',     # Red
        'pino': '#3B1F2B',         # Dark purple
        'ufno': '#95C623',         # Green
    }
    
    markers = {
        'pi_jepa': 'o',
        'fno': 's',
        'geo_fno': '^',
        'deeponet': 'D',
        'pino': 'v',
        'ufno': 'p',
    }
    
    labels = {
        'pi_jepa': 'PI-JEPA (ours)',
        'fno': 'FNO',
        'geo_fno': 'Geo-FNO',
        'deeponet': 'DeepONet',
        'pino': 'PINO',
        'ufno': 'U-FNO',
    }
    
    # Plot each model
    for model_name, model_results in results.items():
        if model_name.startswith('_'):  # Skip metadata
            continue
        
        # Sort by n_labeled
        n_labeled_list = sorted(model_results.keys())
        errors = [model_results[n] for n in n_labeled_list]
        
        color = colors.get(model_name, '#666666')
        marker = markers.get(model_name, 'x')
        label = labels.get(model_name, model_name)
        
        # Highlight PI-JEPA
        linewidth = 2.5 if model_name == 'pi_jepa' else 1.5
        markersize = 10 if model_name == 'pi_jepa' else 7
        
        ax.plot(
            n_labeled_list, errors,
            color=color,
            marker=marker,
            markersize=markersize,
            linewidth=linewidth,
            label=label
        )
    
    # Formatting
    ax.set_xlabel('Number of Labeled Samples ($N_l$)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    if log_scale:
        ax.set_xscale('log')
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to actual n_labeled values
    all_n_labeled = sorted(set(
        n for model_results in results.values()
        if isinstance(model_results, dict)
        for n in model_results.keys()
        if isinstance(n, (int, float))
    ))
    if all_n_labeled:
        ax.set_xticks(all_n_labeled)
        ax.set_xticklabels([str(n) for n in all_n_labeled])
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved data efficiency curves to: {output_path}")
    return output_path


def plot_improvement_bars(
    results: Dict[str, Dict[int, float]],
    output_path: str = "outputs/data_efficiency/improvement_bars.png",
    title: str = "PI-JEPA Improvement over Baselines"
) -> str:
    """
    Generate bar chart showing PI-JEPA improvement over baselines.
    
    Args:
        results: Dict mapping model name to {n_labeled: relative_l2_error}
        output_path: Path to save the figure
        title: Plot title
        
    Returns:
        Path to saved figure
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return ""
    
    if 'pi_jepa' not in results:
        print("Warning: PI-JEPA results not found, skipping improvement plot")
        return ""
    
    # Calculate improvements
    improvements = {}
    for model_name, model_results in results.items():
        if model_name == 'pi_jepa' or model_name.startswith('_'):
            continue
        if not isinstance(model_results, dict):
            continue
        
        model_improvements = []
        for n_labeled in model_results.keys():
            if n_labeled in results['pi_jepa']:
                pijepa_error = results['pi_jepa'][n_labeled]
                baseline_error = model_results[n_labeled]
                if baseline_error > 0:
                    improvement = (baseline_error - pijepa_error) / baseline_error * 100
                    model_improvements.append(improvement)
        
        if model_improvements:
            improvements[model_name] = np.mean(model_improvements)
    
    if not improvements:
        print("Warning: No improvements to plot")
        return ""
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = list(improvements.keys())
    values = [improvements[m] for m in models]
    
    colors = ['#2E86AB' if v > 0 else '#C73E1D' for v in values]
    
    labels = {
        'fno': 'FNO',
        'geo_fno': 'Geo-FNO',
        'deeponet': 'DeepONet',
        'pino': 'PINO',
        'ufno': 'U-FNO',
    }
    display_labels = [labels.get(m, m) for m in models]
    
    bars = ax.bar(display_labels, values, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{value:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10
        )
    
    ax.set_ylabel('Average Improvement (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved improvement bars to: {output_path}")
    return output_path


def save_results_csv(
    results: Dict[str, Dict[int, float]],
    output_dir: str = "outputs/data_efficiency"
) -> List[str]:
    """
    Save results as CSV files for later figure generation.
    
    Args:
        results: Results dict from data efficiency evaluation
        output_dir: Directory to save CSV files
        
    Returns:
        List of paths to saved CSV files
    """
    import csv
    
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # 1. Save main results table (n_labeled x models)
    main_csv = os.path.join(output_dir, "data_efficiency_results.csv")
    
    # Get all n_labeled values and models
    all_n_labeled = sorted(set(
        n for model_name, model_results in results.items()
        if isinstance(model_results, dict) and not model_name.startswith('_')
        for n in model_results.keys()
        if isinstance(n, (int, float))
    ))
    
    models = [m for m in results.keys() if not m.startswith('_') and isinstance(results[m], dict)]
    
    with open(main_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['n_labeled'] + models)
        # Data rows
        for n in all_n_labeled:
            row = [n]
            for model in models:
                error = results[model].get(n, '')
                row.append(error)
            writer.writerow(row)
    
    print(f"Saved results table to: {main_csv}")
    saved_files.append(main_csv)
    
    # 2. Save improvement percentages
    if 'pi_jepa' in results:
        improvement_csv = os.path.join(output_dir, "improvement_over_baselines.csv")
        
        with open(improvement_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['n_labeled', 'baseline', 'pi_jepa_error', 'baseline_error', 'improvement_pct'])
            
            for model_name, model_results in results.items():
                if model_name == 'pi_jepa' or model_name.startswith('_'):
                    continue
                if not isinstance(model_results, dict):
                    continue
                
                for n_labeled in sorted(model_results.keys()):
                    if n_labeled in results['pi_jepa']:
                        pijepa_error = results['pi_jepa'][n_labeled]
                        baseline_error = model_results[n_labeled]
                        if baseline_error > 0:
                            improvement = (baseline_error - pijepa_error) / baseline_error * 100
                            writer.writerow([n_labeled, model_name, pijepa_error, baseline_error, improvement])
        
        print(f"Saved improvement data to: {improvement_csv}")
        saved_files.append(improvement_csv)
    
    # 3. Save summary statistics
    summary_csv = os.path.join(output_dir, "summary_statistics.csv")
    
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'avg_error', 'min_error', 'max_error', 'avg_improvement_vs_pi_jepa'])
        
        for model_name, model_results in results.items():
            if model_name.startswith('_') or not isinstance(model_results, dict):
                continue
            
            errors = list(model_results.values())
            avg_error = sum(errors) / len(errors) if errors else 0
            min_error = min(errors) if errors else 0
            max_error = max(errors) if errors else 0
            
            # Calculate improvement vs PI-JEPA
            if model_name != 'pi_jepa' and 'pi_jepa' in results:
                improvements = []
                for n in model_results.keys():
                    if n in results['pi_jepa']:
                        baseline_error = model_results[n]
                        pijepa_error = results['pi_jepa'][n]
                        if baseline_error > 0:
                            improvements.append((baseline_error - pijepa_error) / baseline_error * 100)
                avg_improvement = sum(improvements) / len(improvements) if improvements else 0
            else:
                avg_improvement = 0
            
            writer.writerow([model_name, avg_error, min_error, max_error, avg_improvement])
    
    print(f"Saved summary statistics to: {summary_csv}")
    saved_files.append(summary_csv)
    
    return saved_files


def generate_all_visualizations(
    results: Dict[str, Dict[int, float]],
    output_dir: str = "outputs/data_efficiency"
) -> List[str]:
    """
    Generate all visualization plots.
    
    Args:
        results: Results dict from data efficiency evaluation
        output_dir: Directory to save figures
        
    Returns:
        List of paths to saved figures
    """
    figures = []
    
    # Data efficiency curves
    curves_path = os.path.join(output_dir, "data_efficiency_curves.png")
    path = plot_data_efficiency_curves(results, curves_path)
    if path:
        figures.append(path)
    
    # Improvement bars
    bars_path = os.path.join(output_dir, "improvement_bars.png")
    path = plot_improvement_bars(results, bars_path)
    if path:
        figures.append(path)
    
    # Also save linear scale version
    linear_path = os.path.join(output_dir, "data_efficiency_curves_linear.png")
    path = plot_data_efficiency_curves(
        results, linear_path,
        title="Data Efficiency Comparison (Linear Scale)",
        log_scale=False
    )
    if path:
        figures.append(path)
    
    return figures


# ============================================================================
# Pipeline Functions
# ============================================================================

def run_pretraining(
    config_path: str,
    output_dir: str,
    n_unlabeled: Optional[int] = None
) -> str:
    """
    Run self-supervised pretraining phase.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for checkpoints
        n_unlabeled: Override N_u from config (optional)
        
    Returns:
        Path to saved checkpoint
    """
    from pretrain import pretrain, build_model_for_pretraining, build_unlabeled_dataloader
    from pretrain import SelfSupervisedPretrainer
    
    print("\n" + "=" * 60)
    print("PHASE 1: Self-Supervised Pretraining")
    print("=" * 60)
    
    # Load config
    config = load_config(config_path)
    
    # Override n_unlabeled if specified
    if n_unlabeled is not None:
        if "pretraining" not in config:
            config["pretraining"] = {}
        config["pretraining"]["n_unlabeled"] = n_unlabeled
        print(f"Overriding N_u = {n_unlabeled}")
    
    # Run pretraining
    checkpoint_path = pretrain(config_path, output_dir)
    
    print(f"\nPretraining complete. Checkpoint: {checkpoint_path}")
    return checkpoint_path


def run_finetuning(
    pretrain_checkpoint: str,
    config_path: str,
    output_dir: str,
    n_labeled: int = 100,
    freeze_encoder: bool = True
) -> str:
    """
    Run supervised finetuning phase.
    
    Args:
        pretrain_checkpoint: Path to pretraining checkpoint
        config_path: Path to configuration file
        output_dir: Output directory for checkpoints
        n_labeled: Number of labeled samples
        freeze_encoder: Whether to freeze encoder
        
    Returns:
        Path to saved checkpoint
    """
    from finetune import finetune
    
    print("\n" + "=" * 60)
    print("PHASE 2: Supervised Finetuning")
    print("=" * 60)
    print(f"N_labeled: {n_labeled}")
    print(f"Freeze encoder: {freeze_encoder}")
    
    checkpoint_path = finetune(
        pretrain_checkpoint=pretrain_checkpoint,
        config_path=config_path,
        n_labeled=n_labeled,
        output_dir=output_dir,
        freeze_encoder=freeze_encoder
    )
    
    print(f"\nFinetuning complete. Checkpoint: {checkpoint_path}")
    return checkpoint_path


def run_data_efficiency_evaluation(
    pretrain_checkpoint: str,
    config_path: str,
    output_dir: str,
    n_labeled_sweep: Optional[List[int]] = None,
    baselines: Optional[List[str]] = None
) -> Dict[str, Dict[int, float]]:
    """
    Run data efficiency comparison.
    
    Args:
        pretrain_checkpoint: Path to pretraining checkpoint
        config_path: Path to configuration file
        output_dir: Output directory for results
        n_labeled_sweep: List of labeled sample counts
        baselines: List of baseline models to compare
        
    Returns:
        Results dict
    """
    from evaluate_data_efficiency import evaluate_data_efficiency
    
    print("\n" + "=" * 60)
    print("PHASE 3: Data Efficiency Evaluation")
    print("=" * 60)
    
    results = evaluate_data_efficiency(
        pretrain_checkpoint=pretrain_checkpoint,
        config_path=config_path,
        output_dir=output_dir,
        n_labeled_sweep=n_labeled_sweep,
        baselines=baselines
    )
    
    return results


def load_existing_results(
    results_path: str
) -> Optional[Dict[str, Dict[int, float]]]:
    """
    Load existing results from JSON file.
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        Results dict or None if file doesn't exist
    """
    if not os.path.exists(results_path):
        return None
    
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        # Convert string keys back to int
        results = {}
        for model_name, model_results in data.items():
            if model_name.startswith('_'):
                results[model_name] = model_results
            else:
                results[model_name] = {
                    int(k): v for k, v in model_results.items()
                }
        
        return results
    except Exception as e:
        print(f"Warning: Could not load results from {results_path}: {e}")
        return None


# ============================================================================
# Main Pipeline
# ============================================================================

def run_full_pipeline(
    config_path: str = "configs/darcy.yaml",
    output_dir: str = "outputs",
    skip_pretrain: bool = False,
    pretrain_checkpoint: Optional[str] = None,
    n_unlabeled: Optional[int] = None,
    n_labeled_sweep: Optional[List[int]] = None,
    baselines: Optional[List[str]] = None,
    visualize: bool = True,
    skip_evaluation: bool = False
) -> Dict[str, Any]:
    """
    Run the complete self-supervised learning pipeline.
    
    This orchestrates:
    1. Self-supervised pretraining on unlabeled coefficient fields
    2. Data efficiency comparison against baselines
    3. Visualization of results
    
    Args:
        config_path: Path to configuration file
        output_dir: Base output directory
        skip_pretrain: Skip pretraining if checkpoint exists
        pretrain_checkpoint: Use existing pretraining checkpoint
        n_unlabeled: Override N_u from config
        n_labeled_sweep: Override N_l sweep from config
        baselines: Baseline models to compare against
        visualize: Generate visualization plots
        skip_evaluation: Skip data efficiency evaluation
        
    Returns:
        Dict with pipeline results
    """
    print("\n" + "=" * 60)
    print("PI-JEPA Self-Supervised Learning Pipeline")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Create output directories
    pretrain_dir = os.path.join(output_dir, "pretrain")
    finetune_dir = os.path.join(output_dir, "finetune")
    eval_dir = os.path.join(output_dir, "data_efficiency")
    
    os.makedirs(pretrain_dir, exist_ok=True)
    os.makedirs(finetune_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    results = {
        'config_path': config_path,
        'output_dir': output_dir,
        'timestamp': datetime.now().isoformat()
    }
    
    # Phase 0: Ensure 64x64 Darcy data exists
    generated_train = os.path.join("data", "darcy", "darcy_train.pt")
    generated_test = os.path.join("data", "darcy", "darcy_test.pt")
    
    if not (os.path.exists(generated_train) and os.path.exists(generated_test)):
        print("\nGenerating 64x64 Darcy flow dataset...")
        from generate_darcy_data import generate_dataset
        import numpy as np
        
        cfg = load_config(config_path)
        resolution = cfg.get("data", {}).get("grid_size", 64)
        seed = cfg.get("experiment", {}).get("seed", 42)
        
        os.makedirs(os.path.join("data", "darcy"), exist_ok=True)
        
        K_train, p_train = generate_dataset(n_samples=1000, resolution=resolution, seed=seed)
        K_test, p_test = generate_dataset(n_samples=200, resolution=resolution, seed=seed + 1)
        
        torch.save(
            {'x': torch.from_numpy(K_train).float().unsqueeze(1),
             'y': torch.from_numpy(p_train).float().unsqueeze(1),
             'resolution': resolution, 'n_samples': 1000},
            generated_train
        )
        torch.save(
            {'x': torch.from_numpy(K_test).float().unsqueeze(1),
             'y': torch.from_numpy(p_test).float().unsqueeze(1),
             'resolution': resolution, 'n_samples': 200},
            generated_test
        )
        print(f"  Saved {resolution}x{resolution} data to data/darcy/")
    else:
        print(f"\nUsing existing 64x64 data at {generated_train}")
    
    # Phase 1: Pretraining
    if pretrain_checkpoint and os.path.exists(pretrain_checkpoint):
        print(f"\nUsing existing checkpoint: {pretrain_checkpoint}")
        checkpoint_path = pretrain_checkpoint
    elif skip_pretrain:
        # Look for existing checkpoint
        default_checkpoint = os.path.join(pretrain_dir, "checkpoint_best.pt")
        if os.path.exists(default_checkpoint):
            print(f"\nUsing existing checkpoint: {default_checkpoint}")
            checkpoint_path = default_checkpoint
        else:
            print("\nNo existing checkpoint found, running pretraining...")
            checkpoint_path = run_pretraining(config_path, pretrain_dir, n_unlabeled)
    else:
        checkpoint_path = run_pretraining(config_path, pretrain_dir, n_unlabeled)
    
    results['pretrain_checkpoint'] = checkpoint_path
    
    # Phase 2: Data Efficiency Evaluation
    if not skip_evaluation:
        eval_results = run_data_efficiency_evaluation(
            pretrain_checkpoint=checkpoint_path,
            config_path=config_path,
            output_dir=eval_dir,
            n_labeled_sweep=n_labeled_sweep,
            baselines=baselines
        )
        results['data_efficiency'] = eval_results
        
        # Always save CSV data for later figure generation
        print("\n" + "=" * 60)
        print("PHASE 4: Saving Results Data")
        print("=" * 60)
        
        csv_files = save_results_csv(eval_results, eval_dir)
        results['csv_files'] = csv_files
        
        # Phase 5: Visualization (optional)
        if visualize:
            print("\n" + "=" * 60)
            print("PHASE 5: Generating Visualizations")
            print("=" * 60)
            
            figures = generate_all_visualizations(eval_results, eval_dir)
            results['figures'] = figures
    else:
        print("\nSkipping data efficiency evaluation")
        
        # Try to load existing results for visualization
        results_path = os.path.join(eval_dir, "benchmark_comparison.json")
        existing_results = load_existing_results(results_path)
        
        if existing_results:
            results['data_efficiency'] = existing_results
            
            # Save CSV from existing results
            csv_files = save_results_csv(existing_results, eval_dir)
            results['csv_files'] = csv_files
            
            if visualize:
                print("\nGenerating visualizations from existing results...")
                figures = generate_all_visualizations(existing_results, eval_dir)
                results['figures'] = figures
    
    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Pretraining checkpoint: {results.get('pretrain_checkpoint', 'N/A')}")
    
    if 'data_efficiency' in results:
        print("\nData Efficiency Results:")
        for model, model_results in results['data_efficiency'].items():
            if model.startswith('_'):
                continue
            if isinstance(model_results, dict):
                avg_error = sum(model_results.values()) / len(model_results)
                print(f"  {model}: avg error = {avg_error:.4f}")
    
    if 'csv_files' in results:
        print(f"\nSaved {len(results['csv_files'])} CSV file(s) for figure generation")
    
    if 'figures' in results:
        print(f"Generated {len(results['figures'])} visualization(s)")
    
    return results


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end self-supervised learning pipeline for PI-JEPA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_self_supervised_pipeline.py --config configs/darcy.yaml
  
  # Skip pretraining if checkpoint exists
  python scripts/run_self_supervised_pipeline.py --skip-pretrain
  
  # Use existing checkpoint
  python scripts/run_self_supervised_pipeline.py --pretrain-checkpoint outputs/pretrain/checkpoint_best.pt
  
  # Custom labeled sample sweep
  python scripts/run_self_supervised_pipeline.py --n-labeled 10 25 50 100
  
  # Only run specific baselines
  python scripts/run_self_supervised_pipeline.py --baselines fno deeponet
  
  # Generate visualizations only (from existing results)
  python scripts/run_self_supervised_pipeline.py --visualize-only
        """
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        default="configs/darcy.yaml",
        help="Path to configuration file (default: configs/darcy.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for all results (default: outputs)"
    )
    
    # Pretraining options
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pretraining if checkpoint exists"
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=str,
        default=None,
        help="Use existing pretraining checkpoint"
    )
    parser.add_argument(
        "--n-unlabeled",
        type=int,
        default=None,
        help="Override N_u (number of unlabeled samples) from config"
    )
    
    # Finetuning/evaluation options
    parser.add_argument(
        "--n-labeled",
        type=int,
        nargs="+",
        default=None,
        help="Override N_l sweep from config (e.g., --n-labeled 10 25 50 100 250 500)"
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=None,
        help="Baseline models to compare against (default: fno geo_fno deeponet)"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip data efficiency evaluation"
    )
    
    # Visualization options
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Generate visualization plots (default: False, saves CSV data only)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization (deprecated, visualizations are off by default)"
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only generate visualizations from existing results"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle visualization-only mode
    if args.visualize_only:
        eval_dir = os.path.join(args.output_dir, "data_efficiency")
        results_path = os.path.join(eval_dir, "benchmark_comparison.json")
        
        results = load_existing_results(results_path)
        if results is None:
            print(f"Error: No results found at {results_path}")
            sys.exit(1)
        
        print("Generating visualizations from existing results...")
        figures = generate_all_visualizations(results, eval_dir)
        print(f"Generated {len(figures)} visualization(s)")
        return
    
    # Determine visualization setting
    visualize = args.visualize and not args.no_visualize
    
    # Run full pipeline
    results = run_full_pipeline(
        config_path=args.config,
        output_dir=args.output_dir,
        skip_pretrain=args.skip_pretrain,
        pretrain_checkpoint=args.pretrain_checkpoint,
        n_unlabeled=args.n_unlabeled,
        n_labeled_sweep=args.n_labeled,
        baselines=args.baselines,
        visualize=visualize,
        skip_evaluation=args.skip_evaluation
    )
    
    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
