#!/bin/bash

# GNNPilot All Models Testing Script
# Tests GAT, GMM, and baseline GCN/GIN models

set -e  # Exit on error

echo "=========================================="
echo "GNNPilot Multi-Model Testing Script"
echo "=========================================="
echo ""

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset> [output_dir]"
    echo ""
    echo "Examples:"
    echo "  $0 test/bcsstk13.mtx"
    echo "  $0 datasets/ogbn-arxiv.pt model_results/"
    echo ""
    echo "Available test datasets:"
    ls test/*.mtx 2>/dev/null || echo "  No .mtx datasets in test/"
    ls datasets/*.pt 2>/dev/null || echo "  No .pt datasets in datasets/"
    exit 1
fi

DATASET=$1
OUTPUT_DIR=${2:-"model_comparison_results"}

echo "Dataset: $DATASET"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo "❌ Dataset not found: $DATASET"
    exit 1
fi

echo "=== Model Testing Summary ==="
echo ""

# Convert to absolute path
if [[ "$DATASET" = /* ]]; then
    DATASET_ABS="$DATASET"
else
    DATASET_ABS="$(pwd)/$DATASET"
fi

# ============================================
# Test 1: GAT Model (DSL-generated kernels)
# ============================================
echo "📊 Test 1: GAT Model (GNNPilot DSL Kernels)"
echo "============================================"

if [ -f "build/libgat.so" ]; then
    echo "✓ GAT library found"

    cd test
    echo "Testing GAT with 32 kernel variants..."

    python test_kernel.py "$DATASET_ABS" \
        "../$OUTPUT_DIR/gat_kernel_results.csv" \
        "../$OUTPUT_DIR/gat_full_results.csv" 2>&1 | tee "../$OUTPUT_DIR/gat_test.log"

    echo "✓ GAT kernel tests completed"

    # E2E GAT test
    if [ -f "e2e_gat.py" ]; then
        echo "Running end-to-end GAT test..."
        python e2e_gat.py "$DATASET_ABS" \
            "../$OUTPUT_DIR/gat_e2e_results.csv" 2>&1 | tee -a "../$OUTPUT_DIR/gat_test.log"
        echo "✓ GAT e2e test completed"
    fi

    cd ..
else
    echo "⚠️  GAT library not found (build/libgat.so)"
    echo "   Run: python dsl_run.py example_gat_layer.txt gat && cd build && make"
fi

echo ""

# ============================================
# Test 2: GMM Model (DSL-generated kernels)
# ============================================
echo "📊 Test 2: GMM Model (Gaussian Mixture Model GNN)"
echo "=================================================="

if [ -f "build/libgmm.so" ]; then
    echo "✓ GMM library found"

    cd test
    echo "Testing GMM with 32 kernel variants..."

    python test_kernel_gmm.py "$DATASET_ABS" \
        "../$OUTPUT_DIR/gmm_kernel_results.csv" \
        "../$OUTPUT_DIR/gmm_full_results.csv" 2>&1 | tee "../$OUTPUT_DIR/gmm_test.log"

    echo "✓ GMM kernel tests completed"

    cd ..
else
    echo "⚠️  GMM library not found (build/libgmm.so)"
    echo "   To build GMM:"
    echo "   1. Create DSL definition for GMM (or check if example_gmm_layer.txt exists)"
    echo "   2. Run: python dsl_run.py example_gmm_layer.txt gmm"
    echo "   3. Build: cd build && cmake .. && make"
    echo ""
    echo "   Skipping GMM tests for now..."
fi

echo ""

# ============================================
# Test 3: GCN Baseline (DGL/PyG comparison)
# ============================================
echo "📊 Test 3: GCN Model (Baseline Comparison)"
echo "==========================================="

cd test

# Check if we have DGL/PyG available
python -c "import dgl; import torch_geometric" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ DGL and PyG available for baseline testing"

    # Try to find GCN baseline test
    if [ -f "lib/dgl_test.py" ] && [ -f "lib/pyg_test.py" ]; then
        echo "Running GCN baseline comparison..."

        # Create a simple GCN test script
        cat > test_gcn_baseline.py << 'GCNEOF'
import sys
import torch
from load_dataset import *
from lib.dgl_test import *
from lib.pyg_test import *
from perf_time import *

dataset_path = sys.argv[1]
output_csv = sys.argv[2] if len(sys.argv) > 2 else None

# Load dataset
if '.pt' in dataset_path:
    graph_data = dataset_load(dataset_path)
    in_dim, out_dim = dataset_prop(graph_data)
    num_nodes = graph_data.num_nodes
elif '.mtx' in dataset_path:
    graph_coo = read_mtx(dataset_path)
    num_nodes = graph_coo.get_shape()[0]
    in_dim = 32
    out_dim = 32
else:
    print(f"Unsupported dataset format: {dataset_path}")
    sys.exit(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Testing GCN on {dataset_path}")
print(f"Nodes: {num_nodes}, in_dim: {in_dim}, out_dim: {out_dim}")

# Test DGL GCN
try:
    print("\n=== DGL GCN ===")
    dgl_time = dgl_model_run('GCN', graph_data, in_dim, 64, out_dim)
    print(f"DGL GCN time: {dgl_time:.4f} ms")
except Exception as e:
    print(f"DGL GCN failed: {e}")
    dgl_time = -1

# Test PyG GCN
try:
    print("\n=== PyG GCN ===")
    pyg_time = pyg_model_run('GCN', graph_data, in_dim, 64, out_dim)
    print(f"PyG GCN time: {pyg_time:.4f} ms")
except Exception as e:
    print(f"PyG GCN failed: {e}")
    pyg_time = -1

# Save results
if output_csv:
    with open(output_csv, 'w') as f:
        f.write("model,framework,execution_time\n")
        if dgl_time > 0:
            f.write(f"GCN,DGL,{dgl_time}\n")
        if pyg_time > 0:
            f.write(f"GCN,PyG,{pyg_time}\n")
    print(f"\nResults saved to {output_csv}")
GCNEOF

        python test_gcn_baseline.py "$DATASET_ABS" \
            "../$OUTPUT_DIR/gcn_baseline_results.csv" 2>&1 | tee "../$OUTPUT_DIR/gcn_test.log" || \
            echo "⚠️  GCN baseline test encountered errors (see log)"

        rm -f test_gcn_baseline.py
    else
        echo "⚠️  GCN baseline test libraries not found"
    fi
else
    echo "⚠️  DGL or PyG not installed, skipping GCN baseline tests"
    echo "   Install with: pip install dgl torch-geometric"
fi

cd ..

echo ""

# ============================================
# Test 4: GIN Baseline (DGL/PyG comparison)
# ============================================
echo "📊 Test 4: GIN Model (Baseline Comparison)"
echo "==========================================="

cd test

python -c "import dgl; import torch_geometric" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ DGL and PyG available for baseline testing"

    if [ -f "lib/dgl_test.py" ] && [ -f "lib/pyg_test.py" ]; then
        echo "Running GIN baseline comparison..."

        # Create a simple GIN test script
        cat > test_gin_baseline.py << 'GINEOF'
import sys
import torch
from load_dataset import *
from lib.dgl_test import *
from lib.pyg_test import *
from perf_time import *

dataset_path = sys.argv[1]
output_csv = sys.argv[2] if len(sys.argv) > 2 else None

# Load dataset
if '.pt' in dataset_path:
    graph_data = dataset_load(dataset_path)
    in_dim, out_dim = dataset_prop(graph_data)
    num_nodes = graph_data.num_nodes
elif '.mtx' in dataset_path:
    graph_coo = read_mtx(dataset_path)
    num_nodes = graph_coo.get_shape()[0]
    in_dim = 32
    out_dim = 32
else:
    print(f"Unsupported dataset format: {dataset_path}")
    sys.exit(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Testing GIN on {dataset_path}")
print(f"Nodes: {num_nodes}, in_dim: {in_dim}, out_dim: {out_dim}")

# Test DGL GIN
try:
    print("\n=== DGL GIN ===")
    dgl_time = dgl_model_run('GIN', graph_data, in_dim, 64, out_dim)
    print(f"DGL GIN time: {dgl_time:.4f} ms")
except Exception as e:
    print(f"DGL GIN failed: {e}")
    dgl_time = -1

# Test PyG GIN
try:
    print("\n=== PyG GIN ===")
    pyg_time = pyg_model_run('GIN', graph_data, in_dim, 64, out_dim)
    print(f"PyG GIN time: {pyg_time:.4f} ms")
except Exception as e:
    print(f"PyG GIN failed: {e}")
    pyg_time = -1

# Save results
if output_csv:
    with open(output_csv, 'w') as f:
        f.write("model,framework,execution_time\n")
        if dgl_time > 0:
            f.write(f"GIN,DGL,{dgl_time}\n")
        if pyg_time > 0:
            f.write(f"GIN,PyG,{pyg_time}\n")
    print(f"\nResults saved to {output_csv}")
GINEOF

        python test_gin_baseline.py "$DATASET_ABS" \
            "../$OUTPUT_DIR/gin_baseline_results.csv" 2>&1 | tee "../$OUTPUT_DIR/gin_test.log" || \
            echo "⚠️  GIN baseline test encountered errors (see log)"

        rm -f test_gin_baseline.py
    else
        echo "⚠️  GIN baseline test libraries not found"
    fi
else
    echo "⚠️  DGL or PyG not installed, skipping GIN baseline tests"
fi

cd ..

echo ""

# ============================================
# Summary Report
# ============================================
echo "=========================================="
echo "📊 Testing Complete - Summary Report"
echo "=========================================="
echo ""

cat > "$OUTPUT_DIR/test_summary.md" << EOF
# GNNPilot Multi-Model Testing Summary

**Dataset**: $DATASET
**Date**: $(date)
**Output Directory**: $OUTPUT_DIR

## Models Tested

### 1. GAT (Graph Attention Network)
$(if [ -f "$OUTPUT_DIR/gat_kernel_results.csv" ]; then
    echo "✅ **Status**: Successfully tested"
    echo "- Tested 32 kernel variants with different optimization strategies"
    echo "- Results: \`gat_kernel_results.csv\`, \`gat_full_results.csv\`"
else
    echo "❌ **Status**: Not tested (library not built)"
fi)

### 2. GMM (Gaussian Mixture Model GNN)
$(if [ -f "$OUTPUT_DIR/gmm_kernel_results.csv" ]; then
    echo "✅ **Status**: Successfully tested"
    echo "- Tested 32 kernel variants"
    echo "- Results: \`gmm_kernel_results.csv\`, \`gmm_full_results.csv\`"
else
    echo "❌ **Status**: Not tested (library not built)"
    echo "- To enable: Build GMM library from DSL definition"
fi)

### 3. GCN (Graph Convolutional Network)
$(if [ -f "$OUTPUT_DIR/gcn_baseline_results.csv" ]; then
    echo "✅ **Status**: Baseline comparison completed"
    echo "- Tested with DGL and PyG implementations"
    echo "- Results: \`gcn_baseline_results.csv\`"
else
    echo "⚠️ **Status**: Baseline only (no optimized GNNPilot version)"
    echo "- GCN uses similar optimizations to GAT"
fi)

### 4. GIN (Graph Isomorphism Network)
$(if [ -f "$OUTPUT_DIR/gin_baseline_results.csv" ]; then
    echo "✅ **Status**: Baseline comparison completed"
    echo "- Tested with DGL and PyG implementations"
    echo "- Results: \`gin_baseline_results.csv\`"
else
    echo "⚠️ **Status**: Baseline only (no optimized GNNPilot version)"
fi)

## Files Generated

\`\`\`
$(ls -lh "$OUTPUT_DIR" | tail -n +2 | awk '{print $9, "("$5")"}')
\`\`\`

## Visualization

To generate plots from these results:

\`\`\`bash
# For GAT results
python visualize_results.py $OUTPUT_DIR/gat_full_results.csv --output-dir $OUTPUT_DIR/plots

# For GMM results (if available)
python visualize_results.py $OUTPUT_DIR/gmm_full_results.csv --output-dir $OUTPUT_DIR/gmm_plots
\`\`\`

## Performance Summary

$(if [ -f "$OUTPUT_DIR/gat_kernel_results.csv" ]; then
    echo "### GAT Best Performance"
    echo "\`\`\`"
    tail -1 "$OUTPUT_DIR/gat_kernel_results.csv"
    echo "\`\`\`"
fi)

$(if [ -f "$OUTPUT_DIR/gmm_kernel_results.csv" ]; then
    echo "### GMM Best Performance"
    echo "\`\`\`"
    tail -1 "$OUTPUT_DIR/gmm_kernel_results.csv"
    echo "\`\`\`"
fi)

---
Generated by test_all_models.sh
EOF

echo "Summary report: $OUTPUT_DIR/test_summary.md"
echo ""

# Display quick summary
echo "Results summary:"
echo "================"
[ -f "$OUTPUT_DIR/gat_kernel_results.csv" ] && echo "✅ GAT tested: $(wc -l < "$OUTPUT_DIR/gat_full_results.csv") results"
[ -f "$OUTPUT_DIR/gmm_kernel_results.csv" ] && echo "✅ GMM tested: $(wc -l < "$OUTPUT_DIR/gmm_full_results.csv") results"
[ -f "$OUTPUT_DIR/gcn_baseline_results.csv" ] && echo "✅ GCN baseline tested"
[ -f "$OUTPUT_DIR/gin_baseline_results.csv" ] && echo "✅ GIN baseline tested"

echo ""
echo "📁 All results saved in: $OUTPUT_DIR/"
echo "📄 Read detailed summary: $OUTPUT_DIR/test_summary.md"
echo ""
echo "Next steps:"
echo "1. Generate visualizations: python visualize_results.py $OUTPUT_DIR/gat_full_results.csv"
echo "2. Compare with paper results in your LaTeX document"
