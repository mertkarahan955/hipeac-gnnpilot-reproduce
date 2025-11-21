if [ $# -ne 1 ]; then
    echo "[Usage]: build_and_run.sh {input_matrix}"
    exit 1
fi

input_matrix=$1

# Dosyanın tam yolunu al
if [ -f "$input_matrix" ]; then
    full_path=$(realpath "$input_matrix")
elif [ -f "$(pwd)/$input_matrix" ]; then
    full_path="$(pwd)/$input_matrix"
else
    echo "Error: File $input_matrix not found"
    exit 1
fi

rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="/home/mertkarahan/miniforge/envs/gnnpilot3/lib/python3.9/site-packages/torch/" ..
make -j 4

cd ../test
python test_kernel.py "$full_path"

### cuDNN kurma kgerek once