#!/usr/bin/env python3
"""
Visualization script for GNNPilot paper reproduction results.
Generates plots similar to the paper figures from CSV results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_csv_results(csv_file):
    """Load CSV results with flexible format detection"""
    try:
        # Read CSV and infer header
        df = pd.read_csv(csv_file)

        # Check if columns look right
        expected_cols = ['dataset', 'kernel_id', 'execution_time']
        if list(df.columns) == expected_cols:
            df.rename(columns={'kernel_id': 'strategy'}, inplace=True)
        elif 'strategy' not in df.columns and 'kernel_id' not in df.columns:
            # No header, assume format: dataset,strategy/kernel_id,execution_time
            df = pd.read_csv(csv_file, header=None)
            if df.shape[1] == 3:
                df.columns = ['dataset', 'strategy', 'execution_time']
            elif df.shape[1] > 3:
                df.columns = ['dataset', 'strategy', 'execution_time'] + [f'metric_{i}' for i in range(df.shape[1]-3)]

        # Clean and convert data types
        # Remove header row if it got included as data
        df = df[df['strategy'] != 'kernel_id']
        df = df[df['strategy'] != 'strategy']

        # Convert to numeric
        df['strategy'] = pd.to_numeric(df['strategy'], errors='coerce')
        df['execution_time'] = pd.to_numeric(df['execution_time'], errors='coerce')

        # Drop rows with NaN values (likely header remnants)
        df = df.dropna(subset=['strategy', 'execution_time'])

        # Convert strategy to int
        df['strategy'] = df['strategy'].astype(int)

        print(f"Loaded {len(df)} results from {csv_file}")
        print(f"Datasets: {df['dataset'].unique()}")
        print(f"Strategies: {sorted(df['strategy'].unique())}")

        return df

    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_kernel_comparison(df, save_path="kernel_comparison.png"):
    """
    Plot Section 4.4: Auto-tuning results showing best kernel selection
    (Similar to paper's kernel performance comparison)
    """
    plt.figure(figsize=(12, 8))
    
    # Group by dataset and plot execution times for different kernels/strategies
    datasets = df['dataset'].unique()
    
    if len(datasets) == 1:
        # Single dataset - show all kernel performance
        dataset_name = datasets[0]
        subset = df[df['dataset'] == dataset_name]
        
        plt.subplot(2, 2, 1)
        plt.bar(subset['strategy'], subset['execution_time'])
        plt.title(f'Kernel Performance Comparison - {dataset_name}')
        plt.xlabel('Kernel ID')
        plt.ylabel('Execution Time (ms)')
        plt.xticks(rotation=45)
        
        # Highlight best kernel
        best_idx = subset['execution_time'].idxmin()
        best_strategy = subset.loc[best_idx, 'strategy']
        best_time = subset.loc[best_idx, 'execution_time']
        plt.axhline(y=best_time, color='red', linestyle='--', 
                   label=f'Best: Kernel {best_strategy} ({best_time:.3f}ms)')
        plt.legend()
        
        # Strategy analysis subplot
        plt.subplot(2, 2, 2)
        strategy_groups = subset.groupby(subset['strategy'] // 8)['execution_time'].mean()
        plt.bar(range(len(strategy_groups)), strategy_groups.values)
        plt.title('Performance by Strategy Group')
        plt.xlabel('Strategy Group')
        plt.ylabel('Avg Execution Time (ms)')
        
    else:
        # Multiple datasets - show comparison
        plt.subplot(2, 1, 1)
        for dataset in datasets:
            subset = df[df['dataset'] == dataset]
            plt.plot(subset['strategy'], subset['execution_time'], 
                    marker='o', label=dataset, linewidth=2)
        
        plt.title('Kernel Performance Across Datasets')
        plt.xlabel('Kernel/Strategy ID')
        plt.ylabel('Execution Time (ms)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Speedup analysis
    plt.subplot(2, 2, 3)
    if len(datasets) > 1:
        speedups = []
        dataset_names = []
        for dataset in datasets:
            subset = df[df['dataset'] == dataset]
            if len(subset) > 1:
                baseline = subset['execution_time'].max()  # worst performance as baseline
                best = subset['execution_time'].min()      # best performance
                speedup = baseline / best
                speedups.append(speedup)
                dataset_names.append(dataset)
        
        plt.bar(dataset_names, speedups)
        plt.title('Optimization Speedup')
        plt.xlabel('Dataset')
        plt.ylabel('Speedup (x)')
        plt.xticks(rotation=45)
        
        # Add speedup values on bars
        for i, v in enumerate(speedups):
            plt.text(i, v + 0.05, f'{v:.2f}x', ha='center', va='bottom')
    
    # Performance distribution
    plt.subplot(2, 2, 4)
    plt.hist(df['execution_time'], bins=20, alpha=0.7, edgecolor='black')
    plt.title('Execution Time Distribution')
    plt.xlabel('Execution Time (ms)')
    plt.ylabel('Frequency')
    plt.axvline(df['execution_time'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["execution_time"].mean():.3f}ms')
    plt.axvline(df['execution_time'].median(), color='green', linestyle='--',
               label=f'Median: {df["execution_time"].median():.3f}ms')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Kernel comparison plot saved to {save_path}")

def plot_strategy_analysis(df, save_path="strategy_analysis.png"):
    """
    Plot Section 3.2 & 3.3: Neighbor packing vs Bin packing analysis
    """
    plt.figure(figsize=(15, 10))
    
    # Analyze different optimization strategies based on kernel IDs
    # Assuming kernel IDs encode different strategies
    df['gather_strategy'] = (df['strategy'] % 2).map({0: 'node-edge', 1: 'edge-based'})
    df['scatter_strategy'] = ((df['strategy'] // 2) % 2).map({0: 'node-edge', 1: 'edge-based'})
    df['dimension_strategy'] = ((df['strategy'] // 4) % 3).map({0: 'node-dim', 1: 'edge-dim', 2: 'node-global-dim'})
    df['fusion_strategy'] = ((df['strategy'] // 12) % 3).map({0: 'no-fusion', 1: 'node-edge', 2: 'all-dim'})
    
    # Gather strategy performance
    plt.subplot(2, 3, 1)
    gather_perf = df.groupby('gather_strategy')['execution_time'].mean()
    plt.bar(gather_perf.index, gather_perf.values, color=['skyblue', 'lightcoral'])
    plt.title('Section 3.2: Gather Strategy Performance')
    plt.xlabel('Gather Strategy')
    plt.ylabel('Avg Execution Time (ms)')
    plt.xticks(rotation=45)
    
    # Add performance numbers on bars
    for i, v in enumerate(gather_perf.values):
        plt.text(i, v + max(gather_perf.values) * 0.01, f'{v:.3f}ms', 
                ha='center', va='bottom')
    
    # Scatter strategy performance
    plt.subplot(2, 3, 2)
    scatter_perf = df.groupby('scatter_strategy')['execution_time'].mean()
    plt.bar(scatter_perf.index, scatter_perf.values, color=['lightgreen', 'orange'])
    plt.title('Section 3.3: Scatter Strategy Performance')
    plt.xlabel('Scatter Strategy')
    plt.ylabel('Avg Execution Time (ms)')
    plt.xticks(rotation=45)
    
    # Fusion strategy performance
    plt.subplot(2, 3, 3)
    fusion_perf = df.groupby('fusion_strategy')['execution_time'].mean()
    colors = ['gold', 'mediumpurple', 'lightpink'][:len(fusion_perf)]
    plt.bar(fusion_perf.index, fusion_perf.values, color=colors)
    plt.title('Section 4.3: Kernel Fusion Performance')
    plt.xlabel('Fusion Strategy')
    plt.ylabel('Avg Execution Time (ms)')
    plt.xticks(rotation=45)
    
    # Dimension strategy performance
    plt.subplot(2, 3, 4)
    dim_perf = df.groupby('dimension_strategy')['execution_time'].mean()
    plt.bar(dim_perf.index, dim_perf.values, color=['cyan', 'magenta', 'yellow'])
    plt.title('Section 4.2: Dimension Parallelization')
    plt.xlabel('Dimension Strategy')
    plt.ylabel('Avg Execution Time (ms)')
    plt.xticks(rotation=45)
    
    # Heatmap of strategy combinations
    plt.subplot(2, 3, 5)
    pivot = df.pivot_table(values='execution_time', 
                          index='gather_strategy', 
                          columns='fusion_strategy', 
                          aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Strategy Combination Heatmap')
    
    # Best strategy identification
    plt.subplot(2, 3, 6)
    best_combinations = df.groupby(['gather_strategy', 'scatter_strategy', 'fusion_strategy'])['execution_time'].mean().sort_values()
    top_5 = best_combinations.head(5)
    
    labels = [f"{g}-{s}-{f}" for (g, s, f) in top_5.index]
    plt.barh(range(len(top_5)), top_5.values, color='lightsteelblue')
    plt.yticks(range(len(top_5)), labels)
    plt.xlabel('Execution Time (ms)')
    plt.title('Top 5 Strategy Combinations')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Strategy analysis plot saved to {save_path}")

def plot_dataset_scalability(df, save_path="dataset_scalability.png"):
    """
    Plot performance scalability across different datasets
    """
    plt.figure(figsize=(12, 8))
    
    datasets = df['dataset'].unique()
    
    if len(datasets) > 1:
        # Performance comparison across datasets
        plt.subplot(2, 2, 1)
        dataset_perf = df.groupby('dataset').agg({
            'execution_time': ['min', 'mean', 'max', 'std']
        }).round(4)
        dataset_perf.columns = ['best', 'average', 'worst', 'std']
        
        x = np.arange(len(datasets))
        width = 0.25
        
        plt.bar(x - width, dataset_perf['best'], width, label='Best', alpha=0.8)
        plt.bar(x, dataset_perf['average'], width, label='Average', alpha=0.8)
        plt.bar(x + width, dataset_perf['worst'], width, label='Worst', alpha=0.8)
        
        plt.xlabel('Dataset')
        plt.ylabel('Execution Time (ms)')
        plt.title('Performance Across Datasets')
        plt.xticks(x, datasets, rotation=45)
        plt.legend()
        
        # Optimization effectiveness
        plt.subplot(2, 2, 2)
        optimization_ratio = dataset_perf['worst'] / dataset_perf['best']
        plt.bar(datasets, optimization_ratio)
        plt.xlabel('Dataset')
        plt.ylabel('Optimization Ratio (worst/best)')
        plt.title('Optimization Effectiveness')
        plt.xticks(rotation=45)
        
        # Add ratio values on bars
        for i, v in enumerate(optimization_ratio):
            plt.text(i, v + 0.1, f'{v:.2f}x', ha='center', va='bottom')
        
        # Best strategy per dataset
        plt.subplot(2, 2, 3)
        best_strategies = df.loc[df.groupby('dataset')['execution_time'].idxmin()]
        plt.scatter(best_strategies['dataset'], best_strategies['strategy'], 
                   s=100, c=best_strategies['execution_time'], cmap='viridis')
        plt.xlabel('Dataset')
        plt.ylabel('Best Strategy ID')
        plt.title('Optimal Strategy per Dataset')
        plt.xticks(rotation=45)
        plt.colorbar(label='Execution Time (ms)')
        
        # Performance variance
        plt.subplot(2, 2, 4)
        plt.bar(datasets, dataset_perf['std'])
        plt.xlabel('Dataset')
        plt.ylabel('Performance Std Dev')
        plt.title('Performance Variance by Dataset')
        plt.xticks(rotation=45)
    
    else:
        # Single dataset analysis
        dataset = datasets[0]
        plt.suptitle(f'Single Dataset Analysis: {dataset}')
        
        strategies = df['strategy'].unique()
        times = df['execution_time'].values
        
        plt.subplot(2, 2, 1)
        # Ensure arrays have same length
        strategies = df['strategy'].values
        times = df['execution_time'].values
        plt.plot(strategies, times, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Strategy/Kernel ID')
        plt.ylabel('Execution Time (ms)')
        plt.title('Performance vs Strategy')
        plt.grid(True, alpha=0.3)
        
        # Highlight best and worst
        best_idx = np.argmin(times)
        worst_idx = np.argmax(times)
        plt.scatter(strategies[best_idx], times[best_idx], color='green', s=100, label='Best')
        plt.scatter(strategies[worst_idx], times[worst_idx], color='red', s=100, label='Worst')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Dataset scalability plot saved to {save_path}")

def generate_performance_report(df, save_path="performance_report.txt"):
    """Generate a text report summarizing the results"""
    with open(save_path, 'w') as f:
        f.write("GNNPilot Reproduction Results - Performance Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Datasets tested: {list(df['dataset'].unique())}\n")
        f.write(f"Strategies/Kernels tested: {len(df['strategy'].unique())}\n\n")
        
        # Overall statistics
        f.write("Overall Performance Statistics:\n")
        f.write(f"  Best execution time: {df['execution_time'].min():.4f} ms\n")
        f.write(f"  Worst execution time: {df['execution_time'].max():.4f} ms\n")
        f.write(f"  Average execution time: {df['execution_time'].mean():.4f} ms\n")
        f.write(f"  Optimization speedup: {df['execution_time'].max() / df['execution_time'].min():.2f}x\n\n")
        
        # Best strategies per dataset
        f.write("Best Strategy per Dataset:\n")
        best_per_dataset = df.loc[df.groupby('dataset')['execution_time'].idxmin()]
        for _, row in best_per_dataset.iterrows():
            f.write(f"  {row['dataset']}: Strategy {row['strategy']} ({row['execution_time']:.4f} ms)\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("End of Report\n")
    
    print(f"Performance report saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize GNNPilot reproduction results')
    parser.add_argument('csv_file', help='CSV file containing results')
    parser.add_argument('--output-dir', default='./plots', help='Output directory for plots')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'], help='Output format')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = load_csv_results(args.csv_file)
    if df is None:
        return
    
    # Generate plots
    output_dir = Path(args.output_dir)
    
    print("\nGenerating visualization plots...")
    
    plot_kernel_comparison(df, output_dir / f"kernel_comparison.{args.format}")
    plot_strategy_analysis(df, output_dir / f"strategy_analysis.{args.format}")
    plot_dataset_scalability(df, output_dir / f"dataset_scalability.{args.format}")
    
    # Generate report
    generate_performance_report(df, output_dir / "performance_report.txt")
    
    print(f"\nAll plots and reports saved to {output_dir}")
    print("\nSummary of generated files:")
    print(f"  - kernel_comparison.{args.format}: Section 4.4 auto-tuning results")
    print(f"  - strategy_analysis.{args.format}: Section 3.2, 3.3, 4.2, 4.3 analysis")
    print(f"  - dataset_scalability.{args.format}: Performance scalability")
    print(f"  - performance_report.txt: Numerical summary")

if __name__ == "__main__":
    # If no arguments provided, show example usage
    if len(__import__('sys').argv) == 1:
        print("Example usage:")
        print("  python visualize_results.py test/test_csv.csv")
        print("  python visualize_results.py results.csv --output-dir ./paper_figures")
        print("  python visualize_results.py full_results.csv --format pdf")
        
        # Try to visualize existing CSV if found
        test_csv = "test/test_csv.csv"
        if os.path.exists(test_csv):
            print(f"\nFound existing CSV file: {test_csv}")
            print("Generating sample visualization...")
            df = load_csv_results(test_csv)
            if df is not None:
                plot_kernel_comparison(df, "sample_visualization.png")
    else:
        main()