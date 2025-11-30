# GNNPilot Reproduction Guide

This guide provides step-by-step instructions to reproduce the GNNPilot paper results from scratch.

## Prerequisites

- Linux OS (tested on Ubuntu 20.04)
- NVIDIA GPU with CUDA support (tested on GTX 1060 6GB)
- Anaconda/Miniconda

## Step 1: Environment Setup

### 1.1 Create Conda Environment

```bash
# Create new environment
conda create -n gnnpilot python=3.9
conda activate gnnpilot
```

### 1.2 Install PyTorch with CUDA 11.3/11.4

```bash
# PyTorch 1.10.2 with CUDA 11.3 (compatible with CUDA 11.4)
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### 1.3 Install MKL (Math Kernel Library)

```bash
# Intel MKL 2022.2.1
conda install mkl=2022.2.1 mkl-include -c conda-forge
```

### 1.4 Install Graph Libraries

```bash
# DGL 1.0.1 with CUDA 11.3
pip install dgl-cu113==0.9.1.post1 dglgo -f https://data.dgl.ai/wheels/repo.html

# OGB (Open Graph Benchmark)
pip install ogb

# PyTorch Geometric (optional, for baseline comparison)
pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
```

### 1.5 Install Visualization Libraries

```bash
pip install matplotlib seaborn pandas numpy scipy
```

### 1.6 Verify Installation

```bash
python << 'EOF'
import torch
import dgl
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"DGL: {dgl.__version__}")
print("✅ Environment ready!")
EOF
```

## Step 2: Download Repository

```bash
# Clone the repository (or your fork)
git clone https://github.com/USTC-ADA/GNNPilot
cd GNNPilot
```

## Step 3: Build KG_GNN Module (Core CUDA Kernels)

### 3.1 Configure KG_GNN Build

```bash
cd KG_GNN

# Edit cmake.sh to set correct PyTorch path
# Find your PyTorch path first
TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
echo "PyTorch path: $TORCH_PATH"
```

Edit `KG_GNN/cmake.sh` and update line 4:

```bash
# Change this line:
-DCMAKE_PREFIX_PATH=/path/to/your/pytorch \

# To your actual PyTorch path:
-DCMAKE_PREFIX_PATH=$TORCH_PATH \
```


### 3.2 Build KG_GNN

```bash
# Run CMake configuration
chmod +x cmake.sh
./cmake.sh

# Build the library
cd build
make -j4

# Verify build
ls -lh libKGGNN.so
cd ../..
```

**Expected output:** `build/libKGGNN.so` (shared library file)

## Step 4: Generate DSL Kernels for GAT

### 4.1 Run DSL Code Generator

```bash
# From repository root
python dsl_run.py example_gat_layer.txt gat
```

This generates:
- `gat.cu` - CUDA kernel implementations (32 variants)
- `CMakeLists.txt` - Build configuration

### 4.2 Build GAT Library

```bash
# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure CMake with correct paths
TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
cmake -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
      -DCUDA_ARCH=61 \
      ..

# Note: Adjust CUDA_ARCH for your GPU
# GTX 1060: 61
# RTX 3080: 86

# Build
make -j4

# Verify build
ls -lh libgat.so
cd ..
```

**Expected output:** `build/libgat.so`

## Step 5: Download Datasets

```bash
# Download OGB datasets
python download_datasets.py

# This creates ./datasets/ with:
# - ogbn-arxiv
# - ogbn-proteins
```

## Step 6: Run Tests

### 6.1 Quick Test (Small Dataset)

```bash
# Test with small matrix from test directory
cd test
python test_kernel.py ../test/bcsstk13.mtx test_results.csv test_full.csv
cd ..
```

### 6.2 Full GAT Reproduction

```bash
# Run complete test suite
./run_full_reproduction.sh datasets/ogbn-arxiv.pt reproduction_results/
```

This will:
1. Test all 32 DSL-generated kernel variants
2. Run end-to-end GAT tests
3. Run neighbor packing strategy tests (if KG_GNN available)
4. Generate visualizations
5. Create summary report

### 6.3 Manual Testing

```bash
cd test

# Test specific kernels
python test_kernel.py <dataset> <output_csv> <full_output_csv>

# Example:
python test_kernel.py ../datasets/ogbn-arxiv.pt results.csv full_results.csv

# End-to-end GAT
python e2e_gat.py ../datasets/ogbn-arxiv.pt e2e_results.csv
```

## Step 7: Generate Visualizations

```bash
# Generate paper-style plots
python visualize_results.py reproduction_results/kernel_full_results.csv \
    --output-dir reproduction_results/plots \
    --format png
```

Generated plots:
- `kernel_comparison.png` - Section 4.4 auto-tuning results
- `strategy_analysis.png` - Section 3.2, 3.3, 4.2, 4.3 analysis
- `dataset_scalability.png` - Performance scalability
- `performance_report.txt` - Numerical summary

## Common Issues and Solutions

### Issue: `preprocessing.cu` not found

**Solution:** The file was missing in early repository versions. Contact authors or check updated repository.

### Issue: Undefined symbols during build

**Solution:** Ensure PyTorch path is correctly set in CMakeLists.txt:

```bash
# Check current PyTorch path
python -c "import torch; print(torch.__path__[0])"

# Update CMakeLists.txt or cmake command
```

### Issue: CUDA version mismatch

**Solution:** Verify CUDA and PyTorch compatibility:

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
# CUDA 11.4 → use PyTorch with cu113
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Issue: `libcudart.so.11.0` not found

**Solution:** Add CUDA lib to LD_LIBRARY_PATH:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
```

### Issue: Out of memory on 6GB GPU

**Solution:** Test with smaller datasets or reduce feature dimensions in test scripts.

## Reproduction Timeline

Based on our experience:

- **Environment setup**: 2-4 hours (including troubleshooting)
- **Dependency resolution**: 8-16 hours (finding compatible versions)
- **Build process**: 1-2 hours (after dependencies resolved)
- **Testing and validation**: 4-8 hours
- **Total**: ~40-60 person-hours for complete reproduction

## Expected Results

### GAT on ogbn-arxiv

- **Best kernel runtime**: ~3.5-4.0 ms
- **Worst kernel runtime**: ~8.5-9.0 ms
- **Speedup**: ~2.3-2.5x between worst and best
- **Best strategy**: Typically kernel 16-20 (varies by graph)

### Performance Metrics

- Execution time measured in milliseconds
- 32 kernel variants tested (different gather/scatter/fusion strategies)
- Auto-tuning selects best kernel automatically

## Verification Checklist

- [ ] Environment installed (PyTorch 1.10.2 + CUDA 11.3/11.4)
- [ ] KG_GNN module built (`KG_GNN/build/libKGGNN.so`)
- [ ] GAT library built (`build/libgat.so`)
- [ ] Datasets downloaded (`datasets/ogbn-arxiv.pt`)
- [ ] Test kernel script runs successfully
- [ ] Visualizations generated
- [ ] Results match paper trends (2-3x speedup)

## Architecture-Specific Notes

### For Different GPUs

Update `CUDA_ARCH` in build command:

```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Common values:
# GTX 1060/1070/1080: 61
# RTX 2060/2070/2080: 75
# RTX 3060/3070/3080/3090: 86
# A100: 80

cmake -DCUDA_ARCH=XX ..
```

### For Different CUDA Versions

| CUDA Version | PyTorch Version | Command |
|--------------|-----------------|---------|
| 11.0 | 1.10.2+cu110 | `pip install torch==1.10.2+cu110 -f https://...` |
| 11.1 | 1.10.2+cu111 | `pip install torch==1.10.2+cu111 -f https://...` |
| 11.3/11.4 | 1.10.2+cu113 | `pip install torch==1.10.2+cu113 -f https://...` |
| 10.2 | 1.10.2+cu102 | `pip install torch==1.10.2+cu102 -f https://...` |

## Contact and Support

For reproduction-specific issues:
- Check `CLAUDE.md` for detailed codebase documentation
- See `ENVIRONMENT_SETUP.md` for troubleshooting
- Contact original authors for code-related questions

---

**Last Updated:** December 2024
**Tested Configuration:** GTX 1060 6GB, Ubuntu 20.04, CUDA 11.4, PyTorch 1.10.2
