if [ $# -ne 1 ]; then
    echo "[Usage]: kg_run.sh {input_matrix}"
    exit 1
fi
input_matrix=$1

rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="/home/mertkarahan/miniforge/envs/gnnpilot3/lib/python3.9/site-packages/torch/" ..
make -j 4

cd ../test
python test_kernel.py $input_matrix

### cuDNN kurma kgerek once