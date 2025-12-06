#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(csv_path: Path, label: str):
    if not csv_path.exists():
        print(f"Warning: CSV file not found at {csv_path} (skipping {label})")
        return None
    df = pd.read_csv(csv_path)
    if 'Embedding Dimension' not in df.columns or 'Accuracy (%)' not in df.columns:
        print(f"Warning: CSV at {csv_path} missing required columns (skipping {label})")
        return None
    df = df.sort_values('Embedding Dimension')
    df['label'] = label
    return df


def main():
    base = Path('/home/ubuntu/robert.taylor/matryoshka-vit-compression/runs')

    sources = [
        (base / 'MRL_evals_cifar10' / 'results_summary.csv', 'CIFAR-10 (Joint Ensemble Learning)'),
        (base / 'MRL_evals_cifar100' / 'results_summary.csv', 'CIFAR-100 (Joint Ensemble Learning)'),
        (base / 'MRL_nested_evals_cifar10' / 'results_summary.csv', 'CIFAR-10 (nested MRL)'),
    ]

    data_frames = []
    for path, label in sources:
        df = load_results(path, label)
        if df is not None:
            data_frames.append(df)

    if not data_frames:
        print('No valid CSV files were found. Nothing to plot.')
        return

    all_df = pd.concat(data_frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#2E86AB', '#E74C3C', '#27AE60']

    for (path, label), color in zip(sources, colors):
        df = next((d for d in data_frames if d['label'].iloc[0] == label), None)
        if df is None:
            continue

        linestyle = '--' if 'nested MRL' in label else '-'

        ax.plot(
            df['Embedding Dimension'],
            df['Accuracy (%)'],
            marker='o',
            linewidth=3,
            markersize=10,
            label=label,
            color=color,
            linestyle=linestyle,
        )

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Embedding Dimension', fontsize=32, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=32, fontweight='bold')
    ax.set_title(
        'Accuracy vs Embedding Dimension',
        fontsize=36,
        fontweight='bold',
        pad=20,
    )

    dims = all_df['Embedding Dimension']
    if dims.max() / max(dims.min(), 1) > 100:
        ax.set_xscale('log')
        unique_dims = sorted(dims.unique())
        ax.set_xticks(unique_dims)
        ax.set_xticklabels(unique_dims, fontsize=24)
    else:
        ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)

    ax.legend(fontsize=28)

    plt.tight_layout()

    output_dir = base / 'overlay_plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / 'accuracy_vs_dimension_overlay.png'
    pdf_path = output_dir / 'accuracy_vs_dimension_overlay.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f'Overlay plot saved to: {png_path}')
    print(f'Overlay PDF saved to: {pdf_path}')

    plt.show()


if __name__ == '__main__':
    main()
