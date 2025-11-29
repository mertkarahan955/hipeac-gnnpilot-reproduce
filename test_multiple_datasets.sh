#!/bin/bash

# Test GNNPilot with multiple datasets for scalability analysis

set -e

echo "=========================================="
echo "Multi-Dataset Testing Script"
echo "=========================================="
echo ""

# Output directory
OUTPUT_DIR=${1:-"multi_dataset_results"}
mkdir -p "$OUTPUT_DIR"

echo "Results will be saved to: $OUTPUT_DIR"
echo ""

# Find all available datasets
DATASETS=()

# Check for SuiteSparse matrices
if [ -d "datasets/suitesparse" ]; then
    for f in datasets/suitesparse/*.mtx; do
        [ -f "$f" ] && DATASETS+=("$f")
    done
fi

# Check for test matrices
if [ -d "test" ]; then
    for f in test/*.mtx; do
        [ -f "$f" ] && DATASETS+=("$f")
    done
fi

# Check for OGB datasets
if [ -d "datasets" ]; then
    for f in datasets/*.pt; do
        [ -f "$f" ] && DATASETS+=("$f")
    done
fi

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "❌ No datasets found!"
    echo ""
    echo "Please download datasets first:"
    echo "  python download_suitesparse_datasets.py"
    echo "  python download_datasets.py"
    exit 1
fi

echo "Found ${#DATASETS[@]} datasets:"
for dataset in "${DATASETS[@]}"; do
    echo "  - $dataset"
done
echo ""

# Prepare combined CSV files
COMBINED_CSV="$OUTPUT_DIR/combined_results.csv"
COMBINED_FULL_CSV="$OUTPUT_DIR/combined_full_results.csv"

# Initialize CSV files with headers
echo "dataset,best_kernel,best_time" > "$COMBINED_CSV"
echo "dataset,kernel_id,execution_time" > "$COMBINED_FULL_CSV"

# Test each dataset
echo "=== Starting Tests ==="
echo ""

cd test

SUCCESSFUL=0
FAILED=0

for dataset in "${DATASETS[@]}"; do
    # Get dataset name
    DATASET_NAME=$(basename "$dataset" | sed 's/\.[^.]*$//')

    # Convert to absolute path
    if [[ "$dataset" = /* ]]; then
        DATASET_ABS="$dataset"
    else
        DATASET_ABS="../$dataset"
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $DATASET_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Create output files for this dataset
    DATASET_OUT="../$OUTPUT_DIR/${DATASET_NAME}_best.csv"
    DATASET_FULL="../$OUTPUT_DIR/${DATASET_NAME}_full.csv"

    # Run test
    if python test_kernel.py "$DATASET_ABS" "$DATASET_OUT" "$DATASET_FULL" 2>&1 | tee "../$OUTPUT_DIR/${DATASET_NAME}_log.txt"; then
        echo "✓ Test completed successfully"

        # Append to combined CSV (skip header)
        if [ -f "$DATASET_OUT" ]; then
            tail -n +2 "$DATASET_OUT" >> "../$COMBINED_CSV"
        fi
        if [ -f "$DATASET_FULL" ]; then
            tail -n +2 "$DATASET_FULL" >> "../$COMBINED_FULL_CSV"
        fi

        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "✗ Test failed"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

cd ..

# Generate summary report
echo "=========================================="
echo "Testing Summary"
echo "=========================================="
echo ""
echo "Total datasets: ${#DATASETS[@]}"
echo "Successful: $SUCCESSFUL"
echo "Failed: $FAILED"
echo ""

if [ $SUCCESSFUL -gt 0 ]; then
    echo "Results saved to: $OUTPUT_DIR/"
    echo ""

    # Show quick performance summary
    echo "Performance Summary:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 << EOF
import pandas as pd

try:
    df = pd.read_csv('$COMBINED_CSV')
    print(f"\nDatasets tested: {len(df)}")
    print(f"\nBest overall time: {df['best_time'].min():.3f} ms ({df.loc[df['best_time'].idxmin(), 'dataset']})")
    print(f"Worst overall time: {df['best_time'].max():.3f} ms ({df.loc[df['best_time'].idxmax(), 'dataset']})")
    print(f"Average best time: {df['best_time'].mean():.3f} ms")
    print(f"\nRange: {df['best_time'].min():.3f} - {df['best_time'].max():.3f} ms")
    print(f"Speedup range: {df['best_time'].max() / df['best_time'].min():.2f}x")

    # Show per-dataset summary
    print("\n" + "=" * 50)
    print("Per-Dataset Best Performance:")
    print("=" * 50)
    for idx, row in df.iterrows():
        print(f"{row['dataset']:20s} : {row['best_time']:7.3f} ms (kernel {row['best_kernel']:.0f})")

except Exception as e:
    print(f"Could not generate summary: {e}")
EOF

    echo ""
    echo "Generate visualizations with:"
    echo "  python visualize_results.py $COMBINED_FULL_CSV --output-dir $OUTPUT_DIR/plots"
    echo ""

    # Generate visualizations automatically
    if command -v python &> /dev/null; then
        echo "=== Generating Visualizations ==="
        if python visualize_results.py "$COMBINED_FULL_CSV" --output-dir "$OUTPUT_DIR/plots" 2>&1; then
            echo "✓ Visualizations saved to $OUTPUT_DIR/plots/"
            echo ""
            echo "View plots:"
            ls -lh "$OUTPUT_DIR/plots/"*.png 2>/dev/null || echo "  (No PNG files generated)"
        else
            echo "⚠️  Visualization generation failed (see errors above)"
        fi
    fi
fi

echo ""
echo "All results available in: $OUTPUT_DIR/"
echo ""
echo "Files generated:"
ls -lh "$OUTPUT_DIR/" | grep -E '\.(csv|txt|png)$' || echo "  (empty)"
