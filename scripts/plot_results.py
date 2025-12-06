#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_accuracy_vs_dimension(csv_path, dataset):
    df = pd.read_csv(csv_path)
    df = df.sort_values('Embedding Dimension')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Embedding Dimension'], df['Accuracy (%)'],
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Embedding Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    if dataset == 'cifar100':
        dataset_title = 'CIFAR-100'
    else:
        dataset_title = 'CIFAR-10'
    ax.set_title(f'Model Accuracy vs Embedding Dimension\nViTFixedPosDualPatchNorm on {dataset_title}', 
                 fontsize=14, fontweight='bold', pad=20)
    if df['Embedding Dimension'].max() / df['Embedding Dimension'].min() > 100:
        ax.set_xscale('log')
        ax.set_xticks(df['Embedding Dimension'])
        ax.set_xticklabels(df['Embedding Dimension'])
    for idx, row in df.iterrows():
        ax.annotate(f"{row['Accuracy (%)']:.2f}%", 
                   xy=(row['Embedding Dimension'], row['Accuracy (%)']),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    output_path = csv_path.parent / 'accuracy_vs_dimension.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    output_pdf = csv_path.parent / 'accuracy_vs_dimension.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_pdf}")
    plt.show()
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Best accuracy: {df['Accuracy (%)'].max():.2f}% at dimension {df.loc[df['Accuracy (%)'].idxmax(), 'Embedding Dimension']}")
    print(f"Worst accuracy: {df['Accuracy (%)'].min():.2f}% at dimension {df.loc[df['Accuracy (%)'].idxmin(), 'Embedding Dimension']}")
    print(f"Accuracy range: {df['Accuracy (%)'].max() - df['Accuracy (%)'].min():.2f}%")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Plot Matryoshka-ViT accuracy vs embedding dimension from TensorBoard-extracted CSV")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"], help="dataset whose results to plot (cifar10 or cifar100)")
    parser.add_argument("--nested", action="store_true", help="use nested MRL results directory instead of the default")
    args = parser.parse_args()
    if args.nested:
        csv_path = Path(f'/home/ubuntu/robert.taylor/matryoshka-vit-compression/runs/MRL_nested_evals_{args.dataset}/results_summary.csv')
    else:
        csv_path = Path(f'/home/ubuntu/robert.taylor/matryoshka-vit-compression/runs/MRL_evals_{args.dataset}/results_summary.csv')
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run extract_tensorboard_results.py first to generate the CSV file.")
        return
    plot_accuracy_vs_dimension(csv_path, args.dataset)

if __name__ == '__main__':
    main()
