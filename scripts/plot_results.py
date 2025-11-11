#!/usr/bin/env python3
"""
Plot accuracy vs embedding dimension from results_summary.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_accuracy_vs_dimension(csv_path):
    """Plot accuracy vs embedding dimension."""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Sort by embedding dimension
    df = df.sort_values('Embedding Dimension')
    
    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot accuracy vs embedding dimension
    ax.plot(df['Embedding Dimension'], df['Accuracy (%)'], 
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Labels and title
    ax.set_xlabel('Embedding Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy vs Embedding Dimension\nViTFixedPosDualPatchNorm on CIFAR-10', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis to log scale if dimensions span multiple orders of magnitude
    if df['Embedding Dimension'].max() / df['Embedding Dimension'].min() > 10:
        ax.set_xscale('log')
        ax.set_xticks(df['Embedding Dimension'])
        ax.set_xticklabels(df['Embedding Dimension'])
    
    # Annotate each point with accuracy value
    for idx, row in df.iterrows():
        ax.annotate(f"{row['Accuracy (%)']:.2f}%", 
                   xy=(row['Embedding Dimension'], row['Accuracy (%)']),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    output_path = csv_path.parent / 'accuracy_vs_dimension.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also save as PDF for publication quality
    output_pdf = csv_path.parent / 'accuracy_vs_dimension.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_pdf}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Best accuracy: {df['Accuracy (%)'].max():.2f}% at dimension {df.loc[df['Accuracy (%)'].idxmax(), 'Embedding Dimension']}")
    print(f"Worst accuracy: {df['Accuracy (%)'].min():.2f}% at dimension {df.loc[df['Accuracy (%)'].idxmin(), 'Embedding Dimension']}")
    print(f"Accuracy range: {df['Accuracy (%)'].max() - df['Accuracy (%)'].min():.2f}%")
    print("="*60)

def main():
    # Path to the CSV file
    csv_path = Path('/home/ubuntu/robert.taylor/vision-transformers-cifar10/runs/MRL_evals/results_summary.csv')
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run extract_tensorboard_results.py first to generate the CSV file.")
        return
    
    # Create the plot
    plot_accuracy_vs_dimension(csv_path)

if __name__ == '__main__':
    main()
