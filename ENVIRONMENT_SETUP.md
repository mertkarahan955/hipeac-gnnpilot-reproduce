# GNNPilot Environment Setup Guide

This guide helps you set up the exact environment needed to reproduce the GNNPilot paper results.

## System Requirements

- **OS**: Linux (tested on Ubuntu 18.04+)
- **GPU**: NVIDIA GPU with CUDA support (tested on GTX 1060 6GB)
- **CUDA**: 11.4 (other 11.x versions may work)
- **Python**: 3.7+
- **Conda**: Anaconda or Miniconda

## Quick Setup (Recommended)

```bash
# 1. Run automated setup script
chmod +x setup_environment.sh
./setup_environment.sh

# 2. Download datasets
python download_datasets.py

# 3. Build the project
./build_and_run.sh test/bcsstk13.mtx

# 4. Test all models
chmod +x test_all_models.sh
./test_all_models.sh datasets/ogbn-arxiv.pt
```

## Detailed Setup

### 1. Create Conda Environment

```bash
# Create environment with Python 3.8
conda create -n gnnpilot python=3.8
conda activate gnnpilot
```

### 2. Install PyTorch (Tested Version)

```bash
# PyTorch 1.10.2 with CUDA 11.4 (exact version used in reproduction)
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=11.3 -c pytorch
```

**Alternative CUDA versions:**
- CUDA 11.1: `cudatoolkit=11.1`
- CUDA 10.2: `cudatoolkit=10.2`

### 3. Install MKL (Math Kernel Library)

```bash
# Intel oneAPI MKL (tested with 2022.2.1)
conda install mkl mkl-include -c intel
```

### 4. Install Graph Processing Libraries

#### OGB (Open Graph Benchmark)
```bash
pip install ogb
```

#### DGL (Deep Graph Library)
```bash
# For CUDA 11.x
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html

# For CUDA 10.2
pip install dgl-cu102 dglgo -f https://data.dgl.ai/wheels/repo.html

# CPU only
pip install dgl
```

#### PyTorch Geometric (PyG)
```bash
# For PyTorch 1.10.2 + CUDA 11.3
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-1.10.2+cu113.html

# For other versions, check: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

### 5. Install Visualization Libraries

```bash
pip install matplotlib seaborn pandas numpy scipy
```

### 6. Optional: Rabbit Reordering Library

```bash
# May not be available on PyPI - usually needs manual installation
pip install rabbit-order || echo "Rabbit not available (optional)"
```

## Verification

Run the verification script to check all dependencies:

```bash
python << 'EOF'
import torch
import dgl
import torch_geometric
from ogb.nodeproppred import NodePropPredDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ DGL: {dgl.__version__}")
print(f"✅ PyG: {torch_geometric.__version__}")
print(f"✅ All dependencies OK!")
EOF
```

## Tested Environment Configuration

This is the exact configuration used for successful reproduction:

```
OS: Ubuntu 20.04 LTS
GPU: NVIDIA GeForce GTX 1060 6GB (Compute Capability 6.1)
CUDA: 11.4
Python: 3.8.13
PyTorch: 1.10.2+cu113
DGL: 0.9.1
PyG: 2.0.4
OGB: 1.3.4
MKL: 2022.2.1
```

## Model-Specific Requirements

### GAT (Graph Attention Networks)
- ✅ Core requirement - always needed
- Build: `python dsl_run.py example_gat_layer.txt gat`

### GMM (Gaussian Mixture Model GNN)
- Requires: PyTorch Geometric with GMMConv
- Check: `python -c "from torch_geometric.nn import GMMConv"`
- Build: Need GMM DSL definition (check with authors)

### GCN (Graph Convolutional Networks)
- Requires: DGL or PyTorch Geometric
- Used for baseline comparison only

### GIN (Graph Isomorphism Networks)
- Requires: DGL or PyTorch Geometric
- Used for baseline comparison only

## Troubleshooting

### PyTorch CUDA version mismatch
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
# For CUDA 11.4: use cudatoolkit=11.3 (closest available)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

### DGL import error
```bash
# Reinstall DGL with correct CUDA version
pip uninstall dgl
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
```

### PyG installation fails
```bash
# Try installing components individually
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-geometric
```

### MKL linking errors during build
```bash
# Set environment variables
export MKLROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Reinstall MKL
conda install -c intel mkl mkl-include --force-reinstall
```

### GMMConv not found
```bash
# GMMConv is included in newer PyG versions
pip install torch-geometric --upgrade

# Verify
python -c "from torch_geometric.nn import GMMConv; print('GMMConv available')"
```

## Performance Profiling (Optional)

For detailed GPU metrics (L1/L2 cache, occupancy):

### NVIDIA Nsight Compute (ncu)
```bash
# Install from CUDA toolkit
sudo apt-get install cuda-nsight-compute-11-4

# Profile a kernel
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum python test/test_kernel.py
```

### Legacy nvprof (deprecated but works)
```bash
# Available in CUDA 11.x but deprecated
nvprof python test/test_kernel.py test/bcsstk13.mtx
```

### PyTorch Profiler (easier)
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    # Your kernel call here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Dataset Download

```bash
# Download OGB datasets
python download_datasets.py

# Datasets will be saved to: ./datasets/
# - ogbn-arxiv
# - ogbn-proteins
# - ogbn-products (large, ~3GB)
```

## Build System

```bash
# Configure CMake with correct paths
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.__path__[0])')" \
      -DCUDA_ARCH=61 \
      ..

# Build (use appropriate number of cores)
make -j4

# Verify build
ls -lh lib*.so
```

## Testing Workflow

```bash
# 1. Setup environment
./setup_environment.sh

# 2. Build project
./build_and_run.sh test/bcsstk13.mtx

# 3. Run comprehensive tests
./test_all_models.sh datasets/ogbn-arxiv.pt model_results/

# 4. Generate visualizations
python visualize_results.py model_results/gat_full_results.csv \
    --output-dir model_results/plots
```

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| `ImportError: libcudart.so.11.0` | Install matching CUDA toolkit or set `LD_LIBRARY_PATH` |
| `undefined symbol: _ZN2at...` | PyTorch version mismatch - rebuild with correct `CMAKE_PREFIX_PATH` |
| `CUDA out of memory` | Reduce batch size or use smaller dataset |
| `GMMConv not found` | Update PyTorch Geometric: `pip install torch-geometric --upgrade` |
| Build fails with MKL errors | Set `MKLROOT` and reinstall MKL via conda |

## Contact

For environment setup issues specific to this reproduction:
- Check [CLAUDE.md](CLAUDE.md) for detailed build instructions
- See original paper authors for code-specific questions

---

Last updated: December 2024
Tested on: NVIDIA GTX 1060 6GB, Ubuntu 20.04, CUDA 11.4
