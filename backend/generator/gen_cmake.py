
def generate_cmake(name, f):
    f.write("cmake_minimum_required(VERSION 3.14.0)\n\n" \
    "project({} LANGUAGES CXX CUDA)\n\n" \
    "find_package(CUDA REQUIRED)\n" \
    "find_package(Torch REQUIRED)\n" \
    "find_library(TORCH_PYTHON_LIBRARY torch_python PATHS \"${{TORCH_INSTALL_PREFIX}}/lib\")\n" \
    "set(CMAKE_CUDA_FLAGS \"${{CMAKE_CUDA_FLAGS}} -w -rdc=true -gencode=arch=compute_86,code=sm_86 -lcudadevrt\")\n" \
    "include_directories(preprocessing_src)\n"
    "include_directories(/root/anaconda3/envs/gnnadvisor/include/python3.6m)\n" \
    "set(SRC_DIR ${{PROJECT_SOURCE_DIR}})\n" \
    "set(SRC_FILE ${{SRC_DIR}}/{}.cu\n" \
    "${{SRC_DIR}}/preprocessing_src/preprocessing.cu)\n" \
    "add_library({} SHARED ${{SRC_FILE}})\n" \
    "target_link_libraries({} \"${{TORCH_LIBRARIES}}\" \"${{TORCH_PYTHON_LIBRARY}}\")".format(name, name, name, name)
    )
