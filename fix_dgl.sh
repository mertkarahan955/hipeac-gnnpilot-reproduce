#!/bin/bash

# Fix DGL installation issues

echo "=========================================="
echo "DGL Installation Fix Script"
echo "=========================================="
echo ""

# Check current DGL
echo "Current DGL installation:"
python -c "import dgl; print('DGL version:', dgl.__version__)" 2>&1 || echo "DGL not properly installed"
echo ""

# Uninstall all DGL packages
echo "=== Uninstalling existing DGL packages ==="
pip uninstall -y dgl dgl-cu113 dgl-cu111 dgl-cu110 dgl-cu102 dglgo 2>/dev/null || true
echo ""

# Check PyTorch version
echo "=== Checking PyTorch version ==="
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'cpu')" 2>/dev/null)

echo "PyTorch: $TORCH_VERSION"
echo "CUDA: $CUDA_VERSION"
echo ""

# Determine correct DGL version
if [[ $CUDA_VERSION == 11.* ]]; then
    DGL_PACKAGE="dgl -f https://data.dgl.ai/wheels/cu113/repo.html"
    echo "Installing DGL for CUDA 11.x..."
elif [[ $CUDA_VERSION == 10.* ]]; then
    DGL_PACKAGE="dgl -f https://data.dgl.ai/wheels/cu102/repo.html"
    echo "Installing DGL for CUDA 10.x..."
else
    DGL_PACKAGE="dgl"
    echo "Installing DGL CPU version..."
fi

echo ""
echo "=== Installing DGL ==="
pip install $DGL_PACKAGE

echo ""
echo "=== Verifying Installation ==="
python << 'EOF'
try:
    import dgl
    print(f"✓ DGL {dgl.__version__} installed successfully")

    # Test basic functionality
    import torch
    g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    print(f"✓ DGL basic functionality works")
    print(f"  Created graph with {g.num_nodes()} nodes, {g.num_edges()} edges")

except Exception as e:
    print(f"✗ DGL verification failed: {e}")
    exit(1)
EOF

VERIFY_STATUS=$?

echo ""
if [ $VERIFY_STATUS -eq 0 ]; then
    echo "=========================================="
    echo "✅ DGL installation fixed successfully!"
    echo "=========================================="
else
    echo "=========================================="
    echo "⚠️  DGL installation still has issues"
    echo "=========================================="
    echo ""
    echo "Alternative: Remove DGL dependency from test scripts"
    echo ""
    echo "Option 1: Fix test_kernel.py (remove unused DGL import)"
    echo "  sed -i 's/^import dgl$/# import dgl  # Not used/' test/test_kernel.py"
    echo ""
    echo "Option 2: Use minimal test without DGL"
    echo "  Create test_kernel_minimal.py without DGL dependency"
fi
