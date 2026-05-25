#!/usr/bin/env python
"""
Figure 1: Method Overview — PI-JEPA Pipeline Diagram.

Generates a schematic showing the pretraining → fine-tuning pipeline:
- Left: Self-supervised pretraining with SGS corpus, masking, predictor bank
- Right: Supervised fine-tuning with prediction head
- Arrows showing data flow and loss terms

Uses matplotlib with colorblind-accessible palette.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


# Colorblind-accessible palette (Wong, 2011)
COLORS = {
    'encoder': '#0072B2',       # blue
    'predictor': '#E69F00',     # orange
    'decoder': '#009E73',       # green
    'physics': '#CC79A7',       # pink/magenta
    'loss': '#D55E00',          # vermillion
    'data': '#56B4E9',          # sky blue
    'target': '#F0E442',        # yellow
    'background': '#FFFFFF',
    'text': '#000000',
    'arrow': '#555555',
}


def draw_box(ax, x, y, width, height, label, color, fontsize=8, alpha=0.85):
    """Draw a labeled rounded box."""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor='black', linewidth=1.0, alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white' if color in [COLORS['encoder'], COLORS['loss']] else 'black')


def draw_arrow(ax, start, end, color=None, style='->', connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS['arrow']
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style, color=color,
        connectionstyle=connectionstyle,
        linewidth=1.5, mutation_scale=12,
    )
    ax.add_patch(arrow)


def create_method_overview(output_path: str, dpi: int = 300):
    """Create the method overview figure."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.5, 'PI-JEPA: Physics-Informed Joint Embedding Predictive Architecture',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # === Left panel: Pretraining ===
    ax.text(2.5, 5.8, 'Self-Supervised Pretraining', ha='center', fontsize=10,
            fontstyle='italic', color=COLORS['encoder'])

    # SGS Corpus
    draw_box(ax, 0.5, 4.5, 1.8, 0.7, 'SGS Corpus\n(K fields)', COLORS['data'])

    # Encoder
    draw_box(ax, 2.5, 4.5, 1.8, 0.7, 'Fourier Encoder\nf_θ', COLORS['encoder'])

    # Masking
    draw_box(ax, 2.5, 3.2, 1.8, 0.6, 'Block Masking', COLORS['data'], alpha=0.6)

    # Predictor Bank
    draw_box(ax, 2.5, 2.0, 1.8, 0.7, 'Predictor Bank\ng_φ₁...g_φK', COLORS['predictor'])

    # EMA Target
    draw_box(ax, 4.8, 4.5, 1.6, 0.6, 'EMA Target\nEncoder', COLORS['target'])

    # Decoder (training only)
    draw_box(ax, 0.8, 1.0, 1.6, 0.6, 'Decoder\n(train only)', COLORS['decoder'])

    # Physics losses
    draw_box(ax, 2.5, 0.2, 1.8, 0.6, 'Spectral Residual\n+ Latent Flux', COLORS['physics'])

    # Loss
    draw_box(ax, 4.5, 1.0, 1.4, 0.6, 'Total Loss\nL_total', COLORS['loss'])

    # Arrows - pretraining flow
    draw_arrow(ax, (1.4, 4.5), (1.6, 4.5))
    draw_arrow(ax, (2.5, 4.1), (2.5, 3.5))
    draw_arrow(ax, (2.5, 2.9), (2.5, 2.35))
    draw_arrow(ax, (2.5, 1.65), (2.5, 0.5))
    draw_arrow(ax, (1.6, 1.0), (1.6, 0.5), connectionstyle='arc3,rad=-0.2')
    draw_arrow(ax, (3.4, 0.5), (3.8, 1.0))
    draw_arrow(ax, (3.4, 2.0), (3.8, 1.3))
    draw_arrow(ax, (4.8, 4.15), (4.5, 1.3), connectionstyle='arc3,rad=0.3')

    # === Right panel: Fine-tuning ===
    ax.text(8.0, 5.8, 'Supervised Fine-Tuning', ha='center', fontsize=10,
            fontstyle='italic', color=COLORS['predictor'])

    # Separator
    ax.axvline(x=6.2, ymin=0.1, ymax=0.9, color='gray', linestyle='--', alpha=0.5)

    # Pretrained encoder
    draw_box(ax, 7.5, 4.5, 1.8, 0.7, 'Pretrained\nEncoder f_θ', COLORS['encoder'])

    # Prediction head
    draw_box(ax, 7.5, 3.0, 1.8, 0.7, 'Prediction\nHead', COLORS['predictor'])

    # Output
    draw_box(ax, 7.5, 1.5, 1.8, 0.7, 'Predicted\nFields', COLORS['data'])

    # Input
    draw_box(ax, 9.5, 4.5, 1.4, 0.6, 'K field\n(input)', COLORS['data'], alpha=0.6)

    # Labeled data
    draw_box(ax, 9.5, 1.5, 1.4, 0.6, 'Ground\nTruth', COLORS['target'])

    # Arrows - fine-tuning flow
    draw_arrow(ax, (8.8, 4.5), (8.8, 4.5))
    draw_arrow(ax, (7.5, 4.1), (7.5, 3.35))
    draw_arrow(ax, (7.5, 2.65), (7.5, 1.85))

    # Data asymmetry annotation
    ax.annotate('', xy=(0.5, 5.2), xytext=(0.5, 5.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['data'], lw=1.5))
    ax.text(0.5, 5.5, 'FREE\n(~5ms)', ha='center', va='center', fontsize=7,
            color=COLORS['data'])

    ax.annotate('', xy=(9.5, 2.2), xytext=(9.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['loss'], lw=1.5))
    ax.text(9.5, 2.9, 'EXPENSIVE\n(min-hrs)', ha='center', va='center', fontsize=7,
            color=COLORS['loss'])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['encoder'], label='Encoder'),
        mpatches.Patch(facecolor=COLORS['predictor'], label='Predictor/Head'),
        mpatches.Patch(facecolor=COLORS['physics'], label='Physics Loss'),
        mpatches.Patch(facecolor=COLORS['data'], label='Data'),
        mpatches.Patch(facecolor=COLORS['decoder'], label='Decoder (discarded)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              framealpha=0.9, ncol=3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 1: Method Overview")
    parser.add_argument("--output", type=str, default="paper/figures/fig1_method_overview.pdf",
                        help="Output file path")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    args = parser.parse_args()

    create_method_overview(args.output, args.dpi)


if __name__ == "__main__":
    main()
