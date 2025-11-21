rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="/home/mertkarahan/miniforge/envs/gnnpilot_fresh/lib/python3.9/site-packages/torch/" ..
make -j 4