# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reproduction repository for the GNNPilot paper - "A Holistic Framework for High-Performance Graph Neural Network Computations on GPUs". The project implements a high-performance and adaptive GNN optimization framework that includes:

1. **DSL-based Code Generation**: A domain-specific language for defining GNN operations that generates optimized CUDA kernels
2. **KG_GNN Module**: Core CUDA implementations with various optimization strategies (neighbor packing, bin packing, row panel-based fusion)
3. **Performance Testing Framework**: Comprehensive benchmarking tools for evaluating different GNN operators and models

## Prerequisites

- Python 3.x with PyTorch (CUDA support required)
- CUDA Toolkit (≥11.0 recommended)
- CMake (≥3.14)
- C++ compiler with C++14 support
- MKL (Intel Math Kernel Library) - preferably via conda
- cuDNN (mentioned in build scripts)

## Build System

The project uses CMake with two main build configurations:

### DSL-Generated Kernels (Primary Workflow)
```bash
# Generate CUDA code from DSL
python dsl_run.py example_gat_layer.txt gat

# Build generated library
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.__path__[0])')" -DCUDA_ARCH=86 ..
make -j4

# Test (from test directory)
cd ../test
python test_kernel.py ../test/bcsstk13.mtx
```

### KG_GNN Module (Alternative Implementation)
```bash
cd KG_GNN
./cmake.sh  # or ./make.sh for direct nvcc compilation
```

### Quick Build and Test
```bash
# Full build and test pipeline
./build_and_run.sh test/bcsstk13.mtx

# Test only (requires existing build)
./run.sh test/bcsstk13.mtx
```

## Architecture

### DSL System (`backend/`)
- **Parser** (`backend/parser/`): Parses DSL files into intermediate representations
  - `KG_parser.py`: Main parsing logic using pyparsing
  - `data_def.py`, `GNNop_def.py`: Define data types and operations
- **Generator** (`backend/generator/`): Converts parsed DSL to optimized CUDA kernels
  - `generate.py`: Main code generation orchestration
  - `gen_kernel.py`: CUDA kernel generation with multiple optimization strategies
  - `gen_cmake.py`: CMakeLists.txt generation for builds

### Core CUDA Implementation (`KG_GNN/core/`)
- **Headers** (`core/include/`): GPU setup, utilities, and main KG_GNN interface
- **Source** (`core/src/`): 
  - `aggregate*.cu`: Various aggregation kernels (SpMM, SDDMM, GAT, GIN)
  - `bin_pack.cu`: Bin packing optimization for data locality
  - `flash_partition.cu`: Memory-efficient partitioning strategies
  - `mm*.cu`: Matrix multiplication pipelines and optimizations
  - `preprocessing.cu`: Graph preprocessing and format conversion

### Testing Framework (`test/`)
- **Core Tests**: `test_kernel.py` - Main kernel testing with performance measurement
- **E2E Tests**: `e2e_gat.py`, `e2e_gmm_*.py` - End-to-end model testing
- **Library Tests** (`test/lib/`): Comparison tests against DGL, PyG, cuSPARSE, etc.
- **Utilities**: `load_dataset.py`, `utils.py`, `perf_time.py`

## DSL Syntax

### Data Types
```
v_data(fd)      # Vertex data with feature dimension fd
e_data(fd)      # Edge data with feature dimension fd  
data(d1, d2)    # Constant data with dimensions d1×d2
```

### Operations
```
gather(OP, "expr")    # Gather operation (SUM, MAX, AVG)
scatter("expr")       # Scatter operation (edge to node)
linear(A, B)          # Matrix multiplication
```

### Indexing
- `i`: Current node index
- `j`: Neighbor node index  
- `nnz`: Edge index
- `:`: All dimensions

## Common Commands

### Code Generation
```bash
# Generate GAT layer implementation
python dsl_run.py example_gat_layer.txt gat

# Generate custom GNN model
python dsl_run.py your_model.txt custom_name
```

### Building
```bash
# Configure build with specific CUDA architecture
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.__path__[0])')" -DCUDA_ARCH=75 ..

# Build with specific number of cores
make -j8
```

### Testing
```bash
# Test with matrix market format
python test/test_kernel.py test/bcsstk13.mtx

# Test specific kernel variants
python test/test_kernel_gat_baseline.py dataset.mtx

# Performance comparison tests
python test/e2e_gat.py dataset.pt
```

### Dataset Management
```bash
# Download OGB datasets
python download_datasets.py

# Load and test with custom datasets
python test/load_dataset.py your_dataset.pt
```

## Key Implementation Details

### Optimization Strategies
- **Neighbor Packing**: Load balancing for sparse graphs
- **Bin Packing**: Improved data locality for dense graphs using BIN_CSR format
- **Row Panel-based Fusion**: Multi-operator optimization with dynamic parallelization
- **Sampling-based Auto-tuning**: Runtime performance optimization

### Generated Kernel Variants
The DSL system generates multiple kernel versions (e.g., `gat_kernel_0` through `gat_kernel_31`) with different:
- Gather strategies: node-based (`n`), node-edge (`ne`), edge-based (`e`)
- Scatter strategies: similar variants
- Dimension strategies: node-dimension (`nd`), edge-dimension (`ed`), node-global-dimension (`ngd`)
- Fusion strategies: no fusion (`no`), node-edge (`ne`), all-dimension (`all_d`)

### Performance Features
- Supports various sparse matrix formats (CSR, COO, BIN_CSR)
- GPU memory optimization with L1 cache reuse
- Multi-GPU support through CUDA streams
- Integration with PyTorch custom operators

## CUDA Architecture Notes

- Default CUDA architecture: `sm_61` (GTX 1060)
- Configurable via `-DCUDA_ARCH=XX` 
- Requires CUDA separable compilation (`-rdc=true`)
- Uses PyTorch's JIT compilation system (`TORCH_LIBRARY`)

## Environment Setup

The build scripts expect specific conda environment paths. Update the `CMAKE_PREFIX_PATH` in build scripts to match your PyTorch installation:
```bash
# Find your PyTorch path
python -c "import torch; print(torch.__path__[0])"
```

This repository implements cutting-edge GNN optimization techniques and serves as a research platform for high-performance graph neural network computations.

## Reproducing Paper Results (Sections 3.2 & 3.3)

### Section 3.2: Neighbor Packing for Sparser Matrices

**Implementation Location**: `KG_GNN/core/src/preprocessing.cu`
- `kg_csr_balance()`: Basic neighbor packing (strategy 1)
- `kg_csr_balance2()`: Enhanced neighbor packing with load balancing (strategy 2)
- `kg_csr_balance3()`: Persistent scheduling for improved L1 cache utilization (strategy 3)
- `kg_csr_balance4()`: Advanced scheduling for maximum L1 hit rate (strategy 4)

**Key Features**:
- Partitions high-degree node neighbors into warp-sized chunks
- Balances workload across warps using parameter `alpha` (typically 1, 5, 10, 15)
- Dynamically adjusts warp size `wsize` based on graph sparsity

**Test Commands**:
```bash
# Test neighbor packing strategies with different baselines
cd test
python test_kernel_gat_baseline.py PCKGNN [dataset.pt/dataset.mtx]
python test_kernel_gmm_baseline.py PCKGNN [dataset.pt/dataset.mtx]

# Compare with other frameworks
python test_kernel_gat_baseline.py UGCG [dataset.pt/dataset.mtx] 
```

### Section 3.3: Bin Packing for Denser Matrices

**Implementation Location**: `KG_GNN/core/src/bin_pack.cu`
- `bin_pack_construct()`: Creates BIN_CSR format for improved data locality
- `sinfo2device()`: Manages bin packing information for GPU execution

**Key Features**:
- Introduces BIN_CSR sparse matrix format
- Groups graph elements into bins for better L1 cache reuse
- Parameters: `bin_size`, `pack_size`, `bin_thresh`, `alpha`
- Separates packed (dense) and sparse parts of the matrix

**Data Structure**:
- `bin_pack` struct: Contains packed and sparse portions
- `PckPtr`, `PckCont`: Packed data pointers and contents
- `RowPtr_sp`, `ColIdx_sp`: Sparse remainder in traditional CSR

**Test Commands**:
```bash
# Test bin packing through baseline scripts (auto-tuning included)
cd test
python test_kernel_gat_baseline.py PCKGNN [dataset.pt/dataset.mtx]
python test_kernel_gmm_baseline.py PCKGNN [dataset.pt/dataset.mtx]

# Compare performance and data locality
python similarity.py [dataset.pt]  # Tests data locality improvements
```

### Auto-tuning Integration

**Location**: `test/lib/kg_test.py` - `kg_autotuning()` function

**Parameters Tested**:
- **Neighbor Packing**: `alpha` ∈ [1, 5, 10, 15], `wsize` ∈ [16, 32, 64, ..., max_degree]
- **Bin Packing**: `bin_size`, `pack_size` based on graph characteristics
- **Strategy Selection**: Automatic choice between strategies 2 and 6

**Evaluation Commands**:
```bash
# Auto-tune and compare strategies (embedded in PCKGNN tests)
python test_kernel_gat_baseline.py PCKGNN ogbn-arxiv.pt results.csv
python test_kernel_gmm_baseline.py PCKGNN ogbn-arxiv.pt results.csv

# Diagnose available operations
python diagnose_ops.py

# Full reproduction pipeline
./e2e_gat_dataset.sh [dataset_directory] [output.csv]
```

### Performance Metrics

The implementation tracks:
- **Execution Time**: Kernel runtime in milliseconds
- **Memory Efficiency**: L1 cache hit rates and data reuse
- **Load Balancing**: Warp utilization across different strategies
- **Strategy Selection**: Optimal parameters for different graph types

### Dataset-Specific Optimization

**Sparse Graphs** (low avg_degree): Favor neighbor packing (strategies 1-4)
**Dense Graphs** (high avg_degree): Favor bin packing (strategy 6)
**Mixed Workloads**: Auto-tuning selects optimal strategy per graph

## Reproducing Paper Visualizations

### Generate Results and Plots
```bash
# Full reproduction pipeline (recommended)
chmod +x run_full_reproduction.sh
./run_full_reproduction.sh test/bcsstk13.mtx results/

# Manual approach
# 1. Run tests and collect CSV results
python test/test_kernel.py test/bcsstk13.mtx results.csv full_results.csv
python test/e2e_gat.py ogbn-arxiv.pt e2e_results.csv

# 2. Generate paper-style visualizations
python visualize_results.py results.csv --output-dir plots/ --format png
```

### Generated Visualizations
The visualization script creates plots matching the paper figures:

- **kernel_comparison.png**: Section 4.4 auto-tuning results
  - Kernel performance comparison across 32 strategies
  - Best kernel identification and speedup analysis
  - Performance distribution and optimization effectiveness

- **strategy_analysis.png**: Section 3.2, 3.3, 4.2, 4.3 analysis  
  - Neighbor packing vs bin packing performance
  - Gather/scatter strategy comparison
  - Kernel fusion effectiveness
  - Dynamic parallelization results

- **dataset_scalability.png**: Performance scalability analysis
  - Cross-dataset performance comparison
  - Optimization ratio per dataset
  - Strategy selection patterns

- **performance_report.txt**: Numerical summary with:
  - Best/worst/average execution times
  - Optimization speedups
  - Optimal strategy per dataset

### Understanding the Results

**32 Kernel Variants** represent combinations of:
- **2** gather strategies (node-edge, edge-based)
- **2** scatter strategies (node-edge, edge-based)  
- **3** dimension strategies (node-dim, edge-dim, node-global-dim)
- **3** fusion strategies (no-fusion, node-edge, all-dim)

**CSV Format**: `dataset,kernel_id,execution_time_ms`
- Lower execution time = better performance
- Best kernel ID indicates optimal strategy combination
- Performance variance shows optimization sensitivity

### Quick Visualization
```bash
# If you already have CSV results
python visualize_results.py test/test_csv.csv

# Compare multiple runs  
python visualize_results.py combined_results.csv --format pdf
```