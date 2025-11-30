#!/usr/bin/env python3
"""
Download OGB (Open Graph Benchmark) datasets for GNNPilot testing.
Downloads all datasets used in the reproduction experiments.
"""

import os
from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset

# Create datasets directory
os.makedirs('./datasets', exist_ok=True)

# Dataset configurations
DATASETS = [
    # Node property prediction datasets
    ('node', 'ogbn-arxiv', 'Node classification on arXiv papers (169K nodes)'),
    ('node', 'ogbn-products', 'Product co-purchasing network (2.4M nodes)'),
    ('node', 'ogbn-proteins', 'Protein-protein association (132K nodes)'),
    ('node', 'ogbn-mag', 'Microsoft Academic Graph (1.9M nodes)'),

    # Link property prediction datasets
    ('link', 'ogbl-collab', 'Author collaboration network (235K nodes)'),
    ('link', 'ogbl-ppa', 'Protein-protein association (576K nodes)'),

    # Reddit is not in OGB, but we'll skip it gracefully
    # ('node', 'reddit', 'Reddit posts dataset'),  # Not in OGB standard
]

def download_dataset(dataset_type, name, description):
    """Download a single OGB dataset"""
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Description: {description}")
    print(f"Type: {dataset_type}")
    print(f"{'='*60}")

    try:
        if dataset_type == 'node':
            dataset = NodePropPredDataset(name=name, root='./datasets')
        elif dataset_type == 'link':
            dataset = LinkPropPredDataset(name=name, root='./datasets')
        else:
            print(f"❌ Unknown dataset type: {dataset_type}")
            return False

        # Get dataset statistics
        print(f"✓ Successfully downloaded {name}")

        # Print basic statistics
        if dataset_type == 'node':
            graph = dataset[0]
            num_nodes = graph[0]['num_nodes']
            num_edges = graph[0]['edge_index'].shape[1]
            print(f"  Nodes: {num_nodes:,}")
            print(f"  Edges: {num_edges:,}")

        return True

    except Exception as e:
        print(f"❌ Failed to download {name}: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("OGB Dataset Downloader for GNNPilot Reproduction")
    print("=" * 60)
    print(f"Download directory: ./datasets/")
    print(f"Total datasets to download: {len(DATASETS)}")
    print()

    success_count = 0
    failed_datasets = []

    for dataset_type, name, description in DATASETS:
        if download_dataset(dataset_type, name, description):
            success_count += 1
        else:
            failed_datasets.append(name)

    # Reddit note
    print(f"\n{'='*60}")
    print("Note: 'reddit' dataset is not available via OGB.")
    print("If needed, download manually from:")
    print("  https://snap.stanford.edu/graphsage/reddit.zip")
    print(f"{'='*60}")

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    print(f"✓ Successful: {success_count}/{len(DATASETS)}")

    if failed_datasets:
        print(f"❌ Failed: {len(failed_datasets)}")
        print(f"   Datasets: {', '.join(failed_datasets)}")

    print()
    print("Downloaded datasets are saved in: ./datasets/")
    print()
    print("To convert to PyTorch format for testing:")
    print("  cd test")
    print("  python load_dataset.py ../datasets/ogbn-arxiv/")
    print()

if __name__ == "__main__":
    main()
