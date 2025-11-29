#!/bin/bash

# GNNPilot Environment Setup Script
# Sets up all required dependencies for reproduction

set -e  # Exit on error

echo "======================================"
echo "GNNPilot Environment Setup"
echo "======================================"
echo ""

# Check conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Detect current environment
CURRENT_ENV=$(conda info --envs | grep '\*' | awk '{print $1}')
echo "Current conda environment: $CURRENT_ENV"
echo ""

# Check PyTorch version
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
echo "Current PyTorch version: $PYTORCH_VERSION"
echo ""

if [[ "$PYTORCH_VERSION" != "1.10.2"* ]]; then
    echo "⚠️  Warning: Expected PyTorch 1.10.2, found $PYTORCH_VERSION"
    echo "The reproduction was tested with PyTorch 1.10.2 + CUDA 11.4"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "=== Installing Required Python Packages ==="
echo ""

# Core dependencies
echo "📦 Installing core dependencies..."
pip install numpy scipy matplotlib seaborn pandas -q

# Graph libraries
echo "📦 Installing OGB (Open Graph Benchmark)..."
pip install ogb -q

echo "📦 Installing DGL (Deep Graph Library)..."
# DGL for CUDA 11.x
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html -q || \
pip install dgl -q

echo "📦 Installing PyTorch Geometric..."
# PyG for PyTorch 1.10.2 + CUDA 11.x
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.2+cu113.html -q || \
pip install torch-geometric -q

# Optional: Rabbit reordering (may not be available)
echo "📦 Attempting to install Rabbit (optional)..."
pip install rabbit-order -q 2>/dev/null || echo "⚠️  Rabbit not available (optional dependency)"

echo ""
echo "=== Verifying Installation ==="
echo ""

# Verification script
python << 'EOF'
import sys

def check_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {display_name:20s} version: {version}")
        return True
    except ImportError:
        print(f"✗ {display_name:20s} NOT INSTALLED")
        return False

print("Required packages:")
all_ok = True
all_ok &= check_import('torch', 'PyTorch')
all_ok &= check_import('numpy', 'NumPy')
all_ok &= check_import('scipy', 'SciPy')
all_ok &= check_import('ogb', 'OGB')
all_ok &= check_import('dgl', 'DGL')

print("\nVisualization packages:")
all_ok &= check_import('matplotlib', 'Matplotlib')
all_ok &= check_import('seaborn', 'Seaborn')
all_ok &= check_import('pandas', 'Pandas')

print("\nPyTorch Geometric:")
try:
    import torch_geometric
    print(f"✓ PyG                version: {torch_geometric.__version__}")

    # Check PyG submodules
    from torch_geometric.nn import GCNConv, GATConv, GINConv
    print("  ✓ GCNConv available")
    print("  ✓ GATConv available")
    print("  ✓ GINConv available")

    try:
        from torch_geometric.nn import GMMConv
        print("  ✓ GMMConv available")
    except ImportError:
        print("  ✗ GMMConv NOT available")
        all_ok = False

except ImportError:
    print("✗ PyG NOT INSTALLED")
    all_ok = False

print("\nOptional packages:")
check_import('rabbit', 'Rabbit')

print("\nCUDA availability:")
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("✗ CUDA not available")
    all_ok = False

sys.exit(0 if all_ok else 1)
EOF

VERIFY_STATUS=$?

echo ""
if [ $VERIFY_STATUS -eq 0 ]; then
    echo "======================================"
    echo "✅ Environment setup completed!"
    echo "======================================"
    echo ""
    echo "You can now run:"
    echo "  ./test_all_models.sh <dataset>"
    echo ""
else
    echo "======================================"
    echo "⚠️  Setup completed with warnings"
    echo "======================================"
    echo ""
    echo "Some packages are missing. Please check the output above."
    echo "You may still be able to run some tests."
    echo ""
fi

# Create datasets directory if it doesn't exist
mkdir -p datasets
echo "📁 Created datasets/ directory"

echo ""
echo "Next steps:"
echo "1. Download datasets: python download_datasets.py"
echo "2. Build the project: ./build_and_run.sh test/bcsstk13.mtx"
echo "3. Run all model tests: ./test_all_models.sh datasets/ogbn-arxiv.pt"
