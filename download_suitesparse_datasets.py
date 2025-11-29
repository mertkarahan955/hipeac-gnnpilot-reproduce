#!/usr/bin/env python3
"""
Download small SuiteSparse matrices for testing different graph characteristics

Selects matrices with diverse properties:
- Small size (< 50K nodes for 6GB GPU)
- Different sparsity patterns
- Various avg_degree values
"""

import os
import urllib.request
import gzip
import shutil

# Create datasets directory
os.makedirs("datasets/suitesparse", exist_ok=True)

# Selected matrices with diverse characteristics
# Format: (group, name, nodes, edges, avg_degree, description)
DATASETS = [
    # Sparse matrices (low avg_degree)
    ("HB", "bcsstk13", 2003, 83883, 41.9, "Structural - very sparse"),
    ("Hamm", "add20", 2395, 13151, 5.5, "Circuit simulation - sparse"),
    ("Newman", "polbooks", 105, 882, 8.4, "Political books network - very sparse"),

    # Medium density
    ("DIMACS10", "delaunay_n10", 1024, 3056, 2.98, "Delaunay triangulation"),
    ("Pajek", "Erdos971", 472, 1314, 2.78, "Erdős collaboration graph"),

    # Denser matrices (higher avg_degree)
    ("SNAP", "ca-GrQc", 5242, 28980, 5.53, "Physics collaboration - medium dense"),
    ("SNAP", "email-Enron", 36692, 367662, 10.02, "Email network - dense"),

    # Different structures
    ("DIMACS10", "caidaRouterLevel", 192244, 1218132, 6.34, "Router network"),
]

def download_matrix(group, name):
    """Download a matrix from SuiteSparse collection"""

    # SuiteSparse URL pattern
    base_url = "https://suitesparse-collection-website.herokuapp.com"
    mtx_url = f"{base_url}/MM/{group}/{name}.tar.gz"

    output_dir = f"datasets/suitesparse/{name}"
    output_file = f"{output_dir}.tar.gz"
    mtx_file = f"datasets/suitesparse/{name}.mtx"

    # Check if already downloaded
    if os.path.exists(mtx_file):
        print(f"✓ {name}.mtx already exists, skipping download")
        return True

    print(f"Downloading {name} from {group}...")

    try:
        # Download tar.gz
        print(f"  Fetching {mtx_url}...")
        urllib.request.urlretrieve(mtx_url, output_file)

        # Extract
        print(f"  Extracting...")
        shutil.unpack_archive(output_file, "datasets/suitesparse/temp")

        # Find .mtx file in extracted directory
        temp_dir = f"datasets/suitesparse/temp/{name}"
        for file in os.listdir(temp_dir):
            if file.endswith(".mtx"):
                src = os.path.join(temp_dir, file)
                # Handle gzipped .mtx files
                if file.endswith(".mtx.gz"):
                    print(f"  Decompressing {file}...")
                    with gzip.open(src, 'rb') as f_in:
                        with open(mtx_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy(src, mtx_file)
                print(f"  ✓ Saved to {mtx_file}")
                break

        # Cleanup
        shutil.rmtree("datasets/suitesparse/temp")
        os.remove(output_file)

        return True

    except Exception as e:
        print(f"  ✗ Error downloading {name}: {e}")
        # Cleanup on error
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists("datasets/suitesparse/temp"):
            shutil.rmtree("datasets/suitesparse/temp")
        return False

def download_from_alternative_source(name):
    """Try alternative download source (direct MTX files)"""

    # Alternative: try NetworkRepository
    alt_url = f"https://networkrepository.com/graphdata/{name}.mtx"
    mtx_file = f"datasets/suitesparse/{name}.mtx"

    try:
        print(f"  Trying alternative source: {alt_url}")
        urllib.request.urlretrieve(alt_url, mtx_file)
        print(f"  ✓ Downloaded from alternative source")
        return True
    except:
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SuiteSparse Matrix Download Script")
    print("=" * 60)
    print(f"\nDownloading {len(DATASETS)} diverse matrices for testing\n")

    success_count = 0
    failed = []

    for group, name, nodes, edges, avg_degree, description in DATASETS:
        print(f"\n[{success_count + 1}/{len(DATASETS)}] {name}")
        print(f"    Nodes: {nodes:,}, Edges: {edges:,}, Avg degree: {avg_degree:.2f}")
        print(f"    Type: {description}")

        # Try main source
        if download_matrix(group, name):
            success_count += 1
        # Try alternative source
        elif download_from_alternative_source(name):
            success_count += 1
        else:
            failed.append(name)
            print(f"  ✗ Failed to download {name}")

    print("\n" + "=" * 60)
    print(f"Download Summary: {success_count}/{len(DATASETS)} successful")
    print("=" * 60)

    if success_count > 0:
        print(f"\n✓ Successfully downloaded {success_count} datasets to datasets/suitesparse/")
        print("\nAvailable datasets:")
        for file in sorted(os.listdir("datasets/suitesparse")):
            if file.endswith(".mtx"):
                size = os.path.getsize(f"datasets/suitesparse/{file}") / 1024
                print(f"  - {file} ({size:.1f} KB)")

    if failed:
        print(f"\n✗ Failed to download: {', '.join(failed)}")
        print("\nManual download options:")
        print("1. Visit https://sparse.tamu.edu/ and search for matrix names")
        print("2. Use test/bcsstk13.mtx (already included)")
        print("3. Generate synthetic matrices with scipy.sparse")

    print("\nNext steps:")
    print("  # Test with multiple datasets")
    print("  ./test_all_models.sh datasets/suitesparse/bcsstk13.mtx results/")
    print("\n  # Or run batch test")
    print("  for f in datasets/suitesparse/*.mtx; do")
    print("    python test/test_kernel.py $f results/$(basename $f .mtx).csv")
    print("  done")
