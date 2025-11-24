if [ $# -ne 1 ]; then
    echo "[Usage]: run.sh {input_matrix}"
    exit 1
fi

input_matrix=$1

if [ -f "$input_matrix" ]; then
    full_path=$(realpath "$input_matrix")
elif [ -f "$(pwd)/$input_matrix" ]; then
    full_path="$(pwd)/$input_matrix"
else
    echo "Error: File $input_matrix not found"
    exit 1
fi

cd test
python test_kernel.py "$full_path"