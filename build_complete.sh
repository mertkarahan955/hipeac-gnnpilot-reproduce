#!/bin/bash

# Complete Build Script for GNNPilot Reproduction
# This script builds everything needed for reproduction in correct order

set -e  # Exit on error

echo "=========================================="
echo "GNNPilot Complete Build Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 0: Check prerequisites
echo "=== Step 0: Checking Prerequisites ==="
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi
echo "✓ Python: $(python --version)"

# Check PyTorch
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${RED}❌ PyTorch not installed${NC}"
    echo "Install with: pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html"
    exit 1
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
echo "✓ PyTorch: $TORCH_VERSION"
echo "✓ CUDA available: $CUDA_AVAILABLE"

# Check DGL
if ! python -c "import dgl" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  DGL not installed (optional for baselines)${NC}"
else
    DGL_VERSION=$(python -c "import dgl; print(dgl.__version__)")
    echo "✓ DGL: $DGL_VERSION"
fi

# Get PyTorch path
TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
echo ""
echo "PyTorch installation path: $TORCH_PATH"
echo ""

# Step 1: Build KG_GNN module
echo "=== Step 1: Building KG_GNN Module ==="
echo ""

if [ ! -d "KG_GNN" ]; then
    echo -e "${RED}❌ KG_GNN directory not found${NC}"
    exit 1
fi

cd KG_GNN

# Update cmake.sh with correct PyTorch path
echo "Updating KG_GNN/cmake.sh with PyTorch path..."
cat > cmake.sh << EOF
rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$TORCH_PATH" ..
make -j 4
EOF

chmod +x cmake.sh

echo "Building KG_GNN..."
./cmake.sh

# Check if build succeeded
if [ ! -f "build/libKGGNN.so" ]; then
    echo -e "${RED}❌ KG_GNN build failed - libKGGNN.so not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ KG_GNN built successfully${NC}"
ls -lh build/libKGGNN.so

cd ..
echo ""

# Step 2: Generate DSL code for GAT
echo "=== Step 2: Generating DSL Code for GAT ==="
echo ""

if [ ! -f "example_gat_layer.txt" ]; then
    echo -e "${RED}❌ example_gat_layer.txt not found${NC}"
    exit 1
fi

echo "Running DSL code generator..."
python dsl_run.py example_gat_layer.txt gat

echo -e "${GREEN}✓ DSL code generated successfully${NC}"


# Step 3: Build GAT library
echo "=== Step 3: Building GAT Library ==="
echo ""

# Create build directory
mkdir -p build
cd build

# Detect GPU architecture
echo "Detecting GPU architecture..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "GPU detected: $GPU_NAME"

    # Set CUDA architecture based on GPU
    if [[ $GPU_NAME == *"1060"* ]] || [[ $GPU_NAME == *"1070"* ]] || [[ $GPU_NAME == *"1080"* ]]; then
        CUDA_ARCH=61
    elif [[ $GPU_NAME == *"2060"* ]] || [[ $GPU_NAME == *"2070"* ]] || [[ $GPU_NAME == *"2080"* ]]; then
        CUDA_ARCH=75
    elif [[ $GPU_NAME == *"3060"* ]] || [[ $GPU_NAME == *"3070"* ]] || [[ $GPU_NAME == *"3080"* ]] || [[ $GPU_NAME == *"3090"* ]]; then
        CUDA_ARCH=86
    elif [[ $GPU_NAME == *"A100"* ]]; then
        CUDA_ARCH=80
    else
        CUDA_ARCH=61  # Default to Pascal
        echo -e "${YELLOW}⚠️  Unknown GPU, using default CUDA_ARCH=61${NC}"
    fi
else
    CUDA_ARCH=61  # Default
    echo -e "${YELLOW}⚠️  nvidia-smi not found, using default CUDA_ARCH=61${NC}"
fi

echo "Using CUDA architecture: sm_$CUDA_ARCH"
echo ""

# Configure CMake
echo "Configuring CMake..."
cmake -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
      -DCUDA_ARCH=$CUDA_ARCH \
      ..

# Build
echo "Building GAT library..."
make -j4

# Check if build succeeded
if [ ! -f "libgat.so" ]; then
    echo -e "${RED}❌ GAT build failed - libgat.so not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ GAT library built successfully${NC}"
ls -lh libgat.so

cd ..
echo ""

# Step 4: Summary
echo "=========================================="
echo -e "${GREEN}✅ Build Complete!${NC}"
echo "=========================================="
echo ""

echo "Built libraries:"
echo "  1. KG_GNN/build/libKGGNN.so"
echo "  2. build/libgat.so"
echo ""

echo "Generated source:"
echo "  - gat.cu (CUDA kernels)"
echo ""

echo "Next steps:"
echo "  1. Download datasets: python download_datasets.py"
echo "  2. Run quick test: cd test && python test_kernel.py ../test/bcsstk13.mtx"
echo "  3. Run full reproduction: ./run_full_reproduction.sh datasets/ogbn-arxiv.pt results/"
echo ""

# Optional: Run quick verification
read -p "Run quick verification test? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== Running Quick Verification ==="
    cd test
    if [ -f "bcsstk13.mtx" ]; then
        python test_kernel.py bcsstk13.mtx verify_results.csv verify_full.csv
        echo -e "${GREEN}✓ Verification test passed${NC}"
        echo "Results saved to: test/verify_results.csv"
    else
        echo -e "${YELLOW}⚠️  Test dataset not found, skipping verification${NC}"
    fi
    cd ..
fi

echo ""
echo "Build script completed successfully!"
