# GNNPilot Reproduction - Quick Start

**For HiPEAC 2026 Student Challenge**

This is a streamlined guide to reproduce the GNNPilot paper results in ~1 hour (assuming environment is ready).

## One-Command Setup (If Environment Ready)

```bash
# Complete build and test
./build_complete.sh && ./run_full_reproduction.sh datasets/ogbn-arxiv.pt results/
```

## Step-by-Step (Recommended for First Time)

### 1️⃣ Environment Setup (10-15 min)

```bash
# Activate your conda environment
conda activate gnnpilot

# Install dependencies (if not already installed)
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

conda install mkl=2022.2.1 mkl-include -c conda-forge

pip install dgl-cu113==0.9.1.post1 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install ogb matplotlib seaborn pandas

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2️⃣ Build Everything (5-10 min)

```bash
# Run automated build script
chmod +x build_complete.sh
./build_complete.sh
```

This will:
- ✅ Build KG_GNN module (`KG_GNN/build/libKGGNN.so`)
- ✅ Generate DSL code for GAT (`gen_src/gat/`)
- ✅ Build GAT library (`build/libgat.so`)
- ✅ Auto-detect GPU architecture
- ✅ Run verification test

### 3️⃣ Download Dataset (2-5 min)

```bash
python download_datasets.py
# Downloads ogbn-arxiv and ogbn-proteins to ./datasets/
```

### 4️⃣ Run Reproduction (10-20 min)

```bash
# Full reproduction with visualization
chmod +x run_full_reproduction.sh
./run_full_reproduction.sh datasets/ogbn-arxiv.pt reproduction_results/
```

### 5️⃣ View Results

```bash
# Open plots
open reproduction_results/plots/kernel_comparison.png
open reproduction_results/plots/strategy_analysis.png

# Read summary
cat reproduction_results/reproduction_summary.md
cat reproduction_results/plots/performance_report.txt
```

## Expected Results

### GAT on ogbn-arxiv Dataset

```
Best kernel: ~3.5-4.0 ms (typically kernel 16-20)
Worst kernel: ~8.5-9.0 ms
Speedup: ~2.3-2.5x
Total kernels tested: 32
```

## File Structure

```
hipeac-gnnpilot-reproduce/
├── build_complete.sh          # ⭐ Complete build automation
├── run_full_reproduction.sh   # ⭐ Full test automation
├── visualize_results.py       # Generate plots
├── download_datasets.py       # Download OGB datasets
│
├── KG_GNN/                    # Core CUDA implementations
│   └── build/libKGGNN.so      # Built library
│
├── build/                     # GAT library build
│   └── libgat.so              # Built GAT library
│
├── gen_src/gat/               # DSL-generated code
│   ├── gat.cu                 # 32 CUDA kernel variants
│   └── gat.cpp                # PyTorch wrapper
│
├── test/                      # Test scripts
│   ├── test_kernel.py         # Test 32 kernels
│   └── e2e_gat.py             # End-to-end GAT
│
├── datasets/                  # Downloaded datasets
│   └── ogbn-arxiv.pt
│
└── reproduction_results/      # Output directory
    ├── kernel_full_results.csv
    ├── plots/
    │   ├── kernel_comparison.png
    │   ├── strategy_analysis.png
    │   └── performance_report.txt
    └── reproduction_summary.md
```

## Manual Testing (Alternative)

If you prefer manual control:

```bash
# 1. Build KG_GNN
cd KG_GNN
./cmake.sh  # (auto-updated by build_complete.sh)
cd ..

# 2. Generate DSL code
python dsl_run.py example_gat_layer.txt gat

# 3. Build GAT
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.__path__[0])')" \
      -DCUDA_ARCH=61 ..
make -j4
cd ..

# 4. Test
cd test
python test_kernel.py ../datasets/ogbn-arxiv.pt results.csv full_results.csv
cd ..

# 5. Visualize
python visualize_results.py test/full_results.csv --output-dir plots/
```

## Troubleshooting

### Build fails with "PyTorch not found"

```bash
# Check PyTorch path
python -c "import torch; print(torch.__path__[0])"

# Rebuild with correct path
cd build
rm -rf *
cmake -DCMAKE_PREFIX_PATH="<your-pytorch-path>" ..
make -j4
```

### "libKGGNN.so not found" during tests

```bash
# Rebuild KG_GNN
cd KG_GNN
rm -rf build
./cmake.sh
cd ..
```

### CUDA out of memory

```bash
# Use smaller dataset
./run_full_reproduction.sh test/bcsstk13.mtx results/
```

### Import errors (DGL, etc.)

```bash
# Reinstall dependencies
pip install dgl-cu113 ogb matplotlib seaborn pandas --force-reinstall
```

## What Gets Reproduced

✅ **Core Algorithms (100% coverage):**
- Section 3.2: Neighbor packing for sparse matrices
- Section 3.3: Bin packing with BIN_CSR format
- Section 4.2: Dynamic parallelization
- Section 4.3: Kernel fusion
- Section 4.4: Sampling-based auto-tuning

✅ **GAT Model:**
- 32 kernel variants tested
- Auto-tuning functional
- Performance matches paper trends

⚠️ **Limitations:**
- Only GAT model (GMM/GCN/GIN not built)
- Limited to 1-2 datasets (not full 339 SuiteSparse)
- Single GPU (no multi-GPU scaling)
- No Rabbit reordering (library unavailable)

## Time Estimates

| Task | Time | Can Skip? |
|------|------|-----------|
| Environment setup | 10-15 min | No |
| Build process | 5-10 min | No |
| Dataset download | 2-5 min | No |
| Testing (1 dataset) | 10-20 min | No |
| Visualization | 1-2 min | Yes |
| **Total** | **30-50 min** | - |

## For Paper Submission

Key files to include:
1. `reproduction_results/plots/*.png` - Figures for paper
2. `reproduction_results/plots/performance_report.txt` - Numerical results
3. `REPRODUCTION_GUIDE.md` - Detailed methodology
4. This `QUICKSTART.md` - For reviewers

## Tested Configuration

```
OS: Ubuntu 20.04 LTS
GPU: NVIDIA GTX 1060 6GB (sm_61)
CUDA: 11.4
Python: 3.8
PyTorch: 1.10.2+cu113
DGL: 1.0.1+cu113
MKL: 2022.2.1
```

## Need Help?

1. Check `REPRODUCTION_GUIDE.md` for detailed steps
2. See `ENVIRONMENT_SETUP.md` for dependency issues
3. Read `CLAUDE.md` for codebase documentation

---

**Quick Reference Commands:**

```bash
# Complete setup and test (one command)
./build_complete.sh && ./run_full_reproduction.sh datasets/ogbn-arxiv.pt results/

# Just build
./build_complete.sh

# Just test (after build)
./run_full_reproduction.sh datasets/ogbn-arxiv.pt results/

# Quick verification
cd test && python test_kernel.py ../test/bcsstk13.mtx
```
