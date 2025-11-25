#!/usr/bin/env python3
"""
Test script for GNNPilot Section 3.2 & 3.3 optimizations
using only existing libraries and test files.
"""

import sys
import torch
import time
from test.load_dataset import *
from test.utils import *
from test.perf_time import *

# Try to load available libraries
libraries_loaded = {}

# Try DSL-generated library
try:
    torch.ops.load_library("build/libgat.so") 
    libraries_loaded['DSL'] = True
    print("✓ DSL-generated library (libgat.so) loaded")
except:
    libraries_loaded['DSL'] = False
    print("✗ DSL-generated library not available")

# Try KG_GNN library
try:
    torch.ops.load_library("KG_GNN/build/libKGGNN.so")
    libraries_loaded['KGGNN'] = True
    print("✓ KG_GNN library (libKGGNN.so) loaded")
    # Import kg_test functions
    sys.path.append('test/lib')
    from kg_test import kg_autotuning, get_ana
except:
    libraries_loaded['KGGNN'] = False
    print("✗ KG_GNN library not available")

def test_neighbor_packing_strategy(dataset_path, device="cuda:0"):
    """Test Section 3.2: Neighbor Packing for Sparser Matrices"""
    print(f"\n=== Testing Neighbor Packing Strategy (Section 3.2) ===")
    print(f"Dataset: {dataset_path}")
    
    if not libraries_loaded['KGGNN']:
        print("KGGNN library required for neighbor packing test")
        return
    
    # Load dataset
    if '.pt' in dataset_path:
        graph_data = dataset_load(dataset_path)
        num_nodes = graph_data.num_nodes
        graph_csr = graph_to_csr(graph_data, reorder=True)
    elif '.mtx' in dataset_path:
        graph_coo = read_mtx(dataset_path)
        num_nodes = graph_coo.get_shape()[0]
        graph_csr = coo_to_csr(num_nodes, graph_coo, reorder=True)
    else:
        print("Unsupported dataset format")
        return
    
    # Calculate graph properties
    m = len(graph_csr.indptr) - 1
    nnz = len(graph_csr.indices)
    avg_rnz = 1.0 * nnz / m
    max_rnz = max(graph_csr.indptr[1:m+1] - graph_csr.indptr[0:m])
    
    print(f"Graph stats: nodes={m}, edges={nnz}, avg_degree={avg_rnz:.2f}, max_degree={max_rnz}")
    
    # Setup tensors
    device = torch.device(device)
    rowptr = torch.tensor(graph_csr.indptr).to(device)
    indices = torch.tensor(graph_csr.indices).to(device)
    input_feature = torch.randn(m, 32).to(device)
    output_feature = torch.zeros(m, 32).to(device)
    
    # Test different neighbor packing strategies
    strategies = {
        "Basic Neighbor Packing (Strategy 1)": 1,
        "Enhanced Load Balancing (Strategy 2)": 2, 
        "Persistent Scheduling (Strategy 3)": 3,
        "L1 Cache Optimization (Strategy 4)": 4
    }
    
    results = {}
    
    for strategy_name, strategy_id in strategies.items():
        print(f"\nTesting {strategy_name}...")
        
        try:
            if strategy_id == 1:
                ana_info = torch.ops.KGGNN.kg_gcn_balance(rowptr, indices, 32)
            elif strategy_id == 2:
                ana_info = torch.ops.KGGNN.kg_gcn_balance2(rowptr, indices, 128, 10)
            elif strategy_id == 3:
                ana_info = torch.ops.KGGNN.kg_gcn_balance3(rowptr, indices, 128, 10)
            elif strategy_id == 4:
                ana_info = torch.ops.KGGNN.kg_gcn_balance4(rowptr, indices, 128)
            
            # Measure performance
            torch.cuda.synchronize()
            start_time = time.time()
            
            torch.ops.KGGNN.kg_gcn_run_balance(rowptr, indices, input_feature, output_feature, ana_info)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            exec_time = (end_time - start_time) * 1000  # ms
            results[strategy_name] = exec_time
            
            print(f"  Execution time: {exec_time:.4f} ms")
            
            # Cleanup
            torch.ops.KGGNN.kg_gcn_finalize(ana_info)
            
        except Exception as e:
            print(f"  Error: {e}")
            results[strategy_name] = None
    
    # Print comparison
    print(f"\n=== Neighbor Packing Results ===")
    best_time = float('inf')
    best_strategy = None
    
    for strategy, exec_time in results.items():
        if exec_time is not None:
            print(f"{strategy}: {exec_time:.4f} ms")
            if exec_time < best_time:
                best_time = exec_time
                best_strategy = strategy
        else:
            print(f"{strategy}: FAILED")
    
    if best_strategy:
        print(f"\nBest strategy: {best_strategy} ({best_time:.4f} ms)")
    
    return results

def test_bin_packing_strategy(dataset_path, device="cuda:0"):
    """Test Section 3.3: Bin Packing for Denser Matrices"""
    print(f"\n=== Testing Bin Packing Strategy (Section 3.3) ===")
    print(f"Dataset: {dataset_path}")
    
    if not libraries_loaded['KGGNN']:
        print("KGGNN library required for bin packing test")
        return
    
    # Load dataset (same as neighbor packing)
    if '.pt' in dataset_path:
        graph_data = dataset_load(dataset_path)
        num_nodes = graph_data.num_nodes
        graph_csr = graph_to_csr(graph_data, reorder=True)
    elif '.mtx' in dataset_path:
        graph_coo = read_mtx(dataset_path)
        num_nodes = graph_coo.get_shape()[0]
        graph_csr = coo_to_csr(num_nodes, graph_coo, reorder=True)
    else:
        print("Unsupported dataset format")
        return
    
    m = len(graph_csr.indptr) - 1
    nnz = len(graph_csr.indices)
    avg_rnz = 1.0 * nnz / m
    
    print(f"Graph stats: nodes={m}, edges={nnz}, avg_degree={avg_rnz:.2f}")
    
    # Setup tensors
    device = torch.device(device)
    rowptr = torch.tensor(graph_csr.indptr).to(device)
    indices = torch.tensor(graph_csr.indices).to(device)
    input_feature = torch.randn(m, 32).to(device)
    output_feature = torch.zeros(m, 32).to(device)
    
    # Test different bin packing configurations
    bin_configs = {
        "Small Bins (256)": (256, 1, 10, 8, 128, 10),
        "Medium Bins (512)": (512, 1, 15, 16, 256, 15),
        "Large Bins (1024)": (1024, 1, 20, 32, 256, 20),
        "Optimized Config": (1024, 256, 20, 32, 64, 10)  # From kg_test.py
    }
    
    results = {}
    
    for config_name, (bin_size, pack_size, bin_thresh, bin_block, wsize, alpha) in bin_configs.items():
        print(f"\nTesting {config_name} (bin_size={bin_size}, wsize={wsize})...")
        
        try:
            ana_info = torch.ops.KGGNN.kg_gcn_bin_pack(rowptr, indices, 
                                                       bin_size, pack_size, bin_thresh, 
                                                       bin_block, wsize, alpha)
            
            # Measure performance
            torch.cuda.synchronize()
            start_time = time.time()
            
            torch.ops.KGGNN.kg_gcn_run_balance(rowptr, indices, input_feature, output_feature, ana_info)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            exec_time = (end_time - start_time) * 1000  # ms
            results[config_name] = exec_time
            
            print(f"  Execution time: {exec_time:.4f} ms")
            
            # Cleanup
            torch.ops.KGGNN.kg_gcn_finalize(ana_info)
            
        except Exception as e:
            print(f"  Error: {e}")
            results[config_name] = None
    
    # Print comparison
    print(f"\n=== Bin Packing Results ===")
    best_time = float('inf')
    best_config = None
    
    for config, exec_time in results.items():
        if exec_time is not None:
            print(f"{config}: {exec_time:.4f} ms")
            if exec_time < best_time:
                best_time = exec_time
                best_config = config
        else:
            print(f"{config}: FAILED")
    
    if best_config:
        print(f"\nBest configuration: {best_config} ({best_time:.4f} ms)")
    
    return results

def test_dsl_kernels(dataset_path, device="cuda:0"):
    """Test DSL-generated kernels (if available)"""
    print(f"\n=== Testing DSL-Generated Kernels ===")
    print(f"Dataset: {dataset_path}")
    
    if not libraries_loaded['DSL']:
        print("DSL library not available")
        return
    
    # This would use the test_kernel.py approach
    print("DSL kernel testing would require test_kernel.py integration")
    # Could integrate the test_kernel.py logic here
    
def main():
    if len(sys.argv) < 2:
        print("Usage: python test_optimization_strategies.py <dataset_path>")
        print("Examples:")
        print("  python test_optimization_strategies.py test/bcsstk13.mtx")
        print("  python test_optimization_strategies.py datasets/ogbn-arxiv.pt")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    print("=== GNNPilot Section 3.2 & 3.3 Optimization Tests ===")
    print(f"Libraries available: {libraries_loaded}")
    
    # Run tests
    if libraries_loaded['KGGNN']:
        neighbor_results = test_neighbor_packing_strategy(dataset_path)
        bin_results = test_bin_packing_strategy(dataset_path)
        
        # Compare neighbor packing vs bin packing
        print(f"\n=== Strategy Comparison ===")
        print("Recommendation based on graph density:")
        
        # Load basic graph stats
        if '.pt' in dataset_path:
            graph_data = dataset_load(dataset_path)
            graph_csr = graph_to_csr(graph_data)
        elif '.mtx' in dataset_path:
            graph_coo = read_mtx(dataset_path)
            num_nodes = graph_coo.get_shape()[0]
            graph_csr = coo_to_csr(num_nodes, graph_coo)
        
        m = len(graph_csr.indptr) - 1
        nnz = len(graph_csr.indices)
        avg_rnz = 1.0 * nnz / m
        
        if avg_rnz < 100:
            print(f"Graph is sparse (avg_degree={avg_rnz:.2f}) -> Neighbor Packing recommended")
        else:
            print(f"Graph is dense (avg_degree={avg_rnz:.2f}) -> Bin Packing recommended")
    
    if libraries_loaded['DSL']:
        test_dsl_kernels(dataset_path)

if __name__ == "__main__":
    main()