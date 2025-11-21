
def generate_cmake(name, f):
    f.write("cmake_minimum_required(VERSION 3.14.0)\n\n" \
    "project({} LANGUAGES CXX CUDA)\n\n" \
    "find_package(CUDA REQUIRED)\n" \
    "find_package(Torch REQUIRED)\n" \
    "find_package(Python3 COMPONENTS Development REQUIRED)\n" \
    "find_library(TORCH_PYTHON_LIBRARY torch_python PATHS \"${{TORCH_INSTALL_PREFIX}}/lib\" \"${{Torch_DIR}}/../lib\" NO_DEFAULT_PATH)\n" \
    "if(NOT TORCH_PYTHON_LIBRARY)\n" \
    "  message(WARNING \"TORCH_PYTHON_LIBRARY not found. Set it manually if needed.\")\n" \
    "endif()\n\n" \
    "# Allow overriding CUDA arch (eg: -DCUDA_ARCH=86)\n" \
    "if(NOT DEFINED CUDA_ARCH)\n" \
    "  set(CUDA_ARCH \"86\")\n" \
    "endif()\n" \
    "set(CMAKE_CUDA_FLAGS \"${{CMAKE_CUDA_FLAGS}} -w -rdc=true -gencode=arch=compute_${{CUDA_ARCH}},code=sm_${{CUDA_ARCH}} -lcudadevrt\")\n\n" \
    "set(SRC_DIR ${{PROJECT_SOURCE_DIR}})\n" \
    "set(PREPROCESSING_DIR ${{SRC_DIR}}/preprocessing_src)\n" \
    "include_directories(${{PREPROCESSING_DIR}})\n" \
    "include_directories(${{Python3_INCLUDE_DIRS}})\n\n" \
    "set(SRC_FILE ${{SRC_DIR}}/{}.cu\n" \
    "${{PREPROCESSING_DIR}}/preprocessing.cu)\n\n" \
    "add_library({} SHARED ${{SRC_FILE}})\n\n" \
    "target_include_directories({} PRIVATE\n" \
    "  ${{PREPROCESSING_DIR}}\n" \
    "  ${{Python3_INCLUDE_DIRS}}\n" \
    "  $<$<BOOL:${{TORCH_INCLUDE_DIRS}}>:${{TORCH_INCLUDE_DIRS}}>\n" \
    ")\n\n" \
    "target_compile_options({} PRIVATE\n" \
    "  $<$<COMPILE_LANGUAGE:CXX>:-I${{Python3_INCLUDE_DIRS}}>\n" \
    "  $<$<COMPILE_LANGUAGE:CUDA>:-I${{Python3_INCLUDE_DIRS}}>\n" \
    ")\n\n" \
    "target_link_libraries({} PRIVATE\n" \
    "  ${{TORCH_LIBRARIES}}\n" \
    "  $<$<BOOL:${{TORCH_PYTHON_LIBRARY}}>:${{TORCH_PYTHON_LIBRARY}}>\n" \
    "  ${{Python3_LIBRARIES}}\n" \
    ")\n\n" \
    "set_target_properties({} PROPERTIES\n" \
    "  CXX_STANDARD 14\n" \
    "  POSITION_INDEPENDENT_CODE ON\n" \
    ")".format(name, name, name, name, name, name, name)
    )
