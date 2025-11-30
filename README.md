# GNNPilot Reproduction Repository

> **A comprehensive reproduction of "GNNPilot: A Holistic Framework for High-Performance Graph Neural Network Computations on GPUs"**
>
> This repository provides a reproducible implementation of the GNNPilot framework with extensive documentation, automated build systems, multi-dataset testing infrastructure, and visualization tools.

**🔗 Original paper:** [GNNPilot](https://github.com/USTC-ADA/GNNPilot)

---

## 🚀 Quick Start

Get up and running with a complete reproduction:

```bash
# 1. Clone repository
git clone https://github.com/mertkarahan955/hipeac-gnnpilot-reproduce.git
cd hipeac-gnnpilot-reproduce

# 2. Setup environment (requires CUDA 11.4)
conda create -n gnnpilot python=3.9
conda activate gnnpilot
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install dgl-cu113 ogb matplotlib seaborn pandas

# 3. One-command build (builds everything: preprocessing, DSL-generated kernels, KG_GNN)
./build_complete.sh

# 4. Download test datasets
python download_suitesparse_datasets.py  # Downloads 9 SuiteSparse graphs

# 5. Run full reproduction pipeline
./run_full_reproduction.sh test/bcsstk13.mtx results/

# 6. Generate publication-quality figures
python visualize_results.py results/combined_full_results.csv --output-dir results/plots
```

**Output:**
- Performance results for 32 kernel variants
- Strategy analysis across gather/scatter/fusion/dimension optimizations
- Auto-tuning speedup analysis
- Publication-ready PNG/PDF figures

---

## 📖 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Core Components](#core-components)
- [Usage Guide](#usage-guide)
- [Multi-Dataset Testing](#multi-dataset-testing)
- [Visualization](#visualization)
- [Implementation Details](#implementation-details)

---

## 🎯 Overview

**GNNPilot** is a  GPU optimization framework for Graph Neural Networks that addresses performance bottlenecks through:

1. **Neighbor Packing** (Section 3.2): Load-balanced sparse operator execution for irregular graphs
2. **Bin Packing** (Section 3.3): Novel BIN_CSR sparse format for improved data locality in dense graphs
3. **Dynamic Parallelization** (Section 4.2): Adaptive thread mapping across node/edge/dimension axes
4. **Row Panel-based Fusion** (Section 4.3): Multi-operator kernel fusion for reduced memory traffic
5. **Sampling-based Auto-tuning** (Section 4.4): Intelligent kernel selection from 32 strategy variants

### What This Repository Provides

This reproduction repository offers:

- ✅ **Complete algorithmic reproduction** of all optimization strategies from the original paper
- ✅ **Automated build system** with dependency management and error handling
- ✅ **Multi-dataset testing infrastructure** supporting 9+ graph datasets (SuiteSparse + OGB)
- ✅ **Publication-quality visualization tools** generating IEEE-format figures
- ✅ **Comprehensive documentation** including detailed reproduction guide and troubleshooting
- ✅ **DSL-based code generation** for automatic CUDA kernel synthesis
- ✅ **Performance benchmarking** against DGL, PyG, cuSPARSE baselines (partial)

### Reproduction Completeness

| Component | Status | Coverage |
|-----------|--------|----------|
| **Neighbor Packing (4 variants)** | ✅ Complete | 100% |
| **Bin Packing (BIN_CSR format)** | ✅ Complete | 100% |
| **Dynamic Parallelization** | ✅ Complete | 100% |
| **Kernel Fusion (3 strategies)** | ✅ Complete | 100% |
| **Auto-tuning (32 kernels)** | ✅ Complete | 100% |
| **GAT Model Implementation** | ✅ Complete | 100% |
| **Multi-dataset Testing** | ✅ Complete | 9 datasets |
| **Visualization Tools** | ✅ Complete | IEEE-format plots |
| **GCN/GIN/GMM Models** | ⚠️ Partial | Infrastructure exists |
| **Multi-GPU Scaling** | ❌ Not implemented | - |
| **Roofline Analysis** | ❌ Hardware limited | - |

## 💻 System Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Tested On |
|-----------|---------|-------------|-----------|
| **GPU** | NVIDIA GPU with CUDA 11.x | RTX 3080+ | GTX 1060 6GB |
| **GPU Memory** | 6 GB | 12 GB+ | 6 GB |
| **CPU** | 4 cores | 8+ cores | Ryzen 5 5600X |
| **RAM** | 8 GB | 16 GB+ | 16 GB |
| **Storage** | 10 GB | 20 GB+ | SSD recommended |

### Software Requirements

| Component | Required Version | Notes |
|-----------|-----------------|-------|
| **Operating System** | Ubuntu 20.04 LTS | May work on 22.04, untested |
| **CUDA Toolkit** | 11.4 | **Exact version required** |
| **Python** | 3.9 | 3.8-3.10 may work |
| **PyTorch** | 1.10.2+cu113 | **Exact version required** |
| **DGL** | 1.0.1 (cu113) | CUDA 11.3 variant |
| **Intel MKL** | 2022.2.1 | Required for preprocessing |
| **GCC** | 9.4.0 | 9.x recommended, 7.x+ may work |
| **CMake** | 3.14+ | For build system |

⚠️ **Version Compatibility Critical:** PyTorch 1.10.2 requires specific CUDA 11.4 + NumPy <1.24 + MKL 2022.x or possıble 2024i.x combination. Newer versions will cause undefined symbol errors.

---

## 🔧 Installation

### Option 1: Automated Conda Setup (Recommended)

```bash
# Create isolated environment
conda create -n gnnpilot python=3.9
conda activate gnnpilot

# Install CUDA-enabled PyTorch (critical: exact version)
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install DGL with CUDA support
pip install dgl-cu113==1.0.1 -f https://data.dgl.ai/wheels/cu113/repo.html

# Install additional dependencies
pip install ogb matplotlib seaborn pandas scipy networkx

# Install Intel MKL (required for preprocessing)
conda install mkl=2022.2.1 mkl-include=2022.2.1

# Verify CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

### Option 2: Manual Installation

If automated setup fails, refer to detailed troubleshooting in [CLAUDE.md](CLAUDE.md) Section "Prerequisites".

### Build All Components

```bash
# Master build script handles:
# 1. Preprocessing library (preprocessing_src/)
# 2. DSL-generated GAT kernels (via dsl_run.py)
# 3. KG_GNN CUDA library (KG_GNN/)
./build_complete.sh

# Expected output:
# ✓ Preprocessing library built → preprocessing_src/preprocessing.so
# ✓ GAT kernels generated and built → build/libgat.so
# ✓ KG_GNN library built → KG_GNN/build/libkg_gnn.so
```

### Download Test Datasets

```bash
# Download 9 SuiteSparse matrices (105 nodes to 192K nodes)
python download_suitesparse_datasets.py

# OR download OGB datasets (larger, 169K-2.4M nodes)
python download_datasets.py

# Datasets saved to test/ directory:
# - test/polbooks.mtx (105 nodes)
# - test/bcsstk13.mtx (2K nodes)
# - test/ogbn-arxiv.pt (169K nodes)
# etc.
```

## 📘 Usage Guide

### Basic Single-Dataset Testing

```bash
# Test GAT kernels on a single dataset
./run_full_reproduction.sh test/bcsstk13.mtx results/

# Outputs:
# - results/kernel_results.csv        # Per-kernel execution times
# - results/combined_full_results.csv # Aggregate results
# - Console: Best kernel ID and speedup
```

### Advanced: Manual Kernel Testing

```bash
# Test specific kernel variants
cd test
python test_kernel.py bcsstk13.mtx results.csv full_results.csv

# Test only gather operations
python test_gather_kernel.py bcsstk13.mtx

# End-to-end GAT model evaluation
python e2e_gat.py ogbn-arxiv.pt
```

### DSL-Based Kernel Generation

```bash
# Generate custom GNN operator kernels
python dsl_run.py example_gat_layer.txt gat

# Build generated kernels
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.__path__[0])')" \
      -DCUDA_ARCH=61 ..  # Use your GPU's compute capability
make -j4
```

**CUDA Architecture Selection:**
- GTX 1060: `-DCUDA_ARCH=61`
- RTX 2080: `-DCUDA_ARCH=75`
- RTX 3080: `-DCUDA_ARCH=86`
- A100: `-DCUDA_ARCH=80`

---

## 🌐 Multi-Dataset Testing

### Batch Testing (9 Datasets)

```bash
# Run all 9 SuiteSparse + OGB datasets
./test_multiple_datasets.sh

# Outputs to multi_dataset_results/:
# - combined_full_results.csv (726 experiments: 9 datasets × 32 kernels × 2.5 runs avg)
# - individual CSV files per dataset
```

### Supported Datasets

| Dataset | Nodes | Edges | Avg Degree | Source |
|---------|-------|-------|------------|--------|
| polbooks | 105 | 441 | 8.4 | SuiteSparse |
| delaunay_n10 | 1,024 | 3,056 | 2.98 | SuiteSparse |
| add20 | 2,395 | 7,462 | 5.5 | SuiteSparse |
| bcsstk13 | 2,003 | 83,883 | 41.9 | SuiteSparse |
| Erdos971 | 472 | 1,314 | 2.78 | SuiteSparse |
| ca-GrQc | 5,242 | 28,980 | 5.53 | SuiteSparse |
| email-Enron | 36,692 | 367,662 | 10.02 | SuiteSparse |
| caidaRouterLevel | 192,244 | 1,218,132 | 6.34 | SuiteSparse |
| ogbn-arxiv | 169,343 | 1,166,243 | 7.9 | OGB |

---

## 📊 Visualization

### Generate Publication Figures

```bash
# Create IEEE-format plots from results
python visualize_results.py multi_dataset_results/combined_full_results.csv \
    --output-dir multi_dataset_results/plots \
    --format png

# Generates:
# 1. kernel_comparison.png - Auto-tuning performance (Section 4.4)
# 2. strategy_analysis_part1.png - Gather & Scatter strategies (Sections 3.2-3.3)
# 3. strategy_analysis_part2.png - Dimension & Fusion strategies (Sections 4.2-4.3)
# 4. strategy_analysis_part3.png - Heatmap & Top-5 combinations
# 5. dataset_scalability.png - Multi-dataset performance trends
# 6. performance_report.txt - Numerical summary
```

### Figure Descriptions

**kernel_comparison.png:**
- Bar chart: Per-kernel execution times across 32 variants
- Line plot: Cross-dataset performance comparison
- Speedup bars: Optimization effectiveness per dataset
- Histogram: Execution time distribution

**strategy_analysis_part1.png (2×2 layout):**
- (a) Gather strategy performance: node-edge vs. edge-based
- (b) Scatter strategy performance: workload distribution comparison

**strategy_analysis_part2.png (2×2 layout):**
- (a) Dimension parallelization: nd vs. ed vs. ngd
- (b) Kernel fusion: no-fusion vs. ne vs. all-dim

**strategy_analysis_part3.png (2×2 layout):**
- (a) Heatmap: Gather×Fusion interaction matrix
- (b) Top-5 combinations: Best-performing strategy tuples

**dataset_scalability.png:**
- Best/average/worst execution times per dataset
- Optimization ratio (worst/best) bars
- Optimal kernel selection scatter plot
- Performance variance analysis