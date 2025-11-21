
def generate_cmake(name, f):
    f.write("cmake_minimum_required(VERSION 3.14.0)\n\n" \
    "project({} LANGUAGES CXX CUDA)\n\n" \
    "find_package(CUDA REQUIRED)\n" \
    "find_package(Torch REQUIRED)\n" \
    "find_package(Python3 COMPONENTS Development REQUIRED)\n" \
    "find_package(Threads REQUIRED)\n" \
    "find_library(TORCH_PYTHON_LIBRARY torch_python PATHS \"${{TORCH_INSTALL_PREFIX}}/lib\" \"${{Torch_DIR}}/../lib\" NO_DEFAULT_PATH)\n" \
    "if(TORCH_PYTHON_LIBRARY)\n" \
    "  message(STATUS \"Found TORCH_PYTHON_LIBRARY: ${{TORCH_PYTHON_LIBRARY}}\")\n" \
    "else()\n" \
    "  message(WARNING \"TORCH_PYTHON_LIBRARY not found automatically. If you get link errors, set TORCH_PYTHON_LIBRARY to the path of torch_python.\")\n" \
    "endif()\n\n" \
    "# --- MKL runtime detection: prefer conda env's libmkl_rt.so if present ---\n" \
    "if(DEFINED ENV{{CONDA_PREFIX}})\n" \
    "  set(CONDA_PREFIX_DIR $ENV{{CONDA_PREFIX}})\n" \
    "else()\n" \
    "  set(CONDA_PREFIX_DIR \"\")\n" \
    "endif()\n\n" \
    "if(CONDA_PREFIX_DIR)\n" \
    "  message(STATUS \"Adding CONDA lib dir to link search: ${{CONDA_PREFIX_DIR}}/lib\")\n" \
    "  link_directories(\"${{CONDA_PREFIX_DIR}}/lib\")\n" \
    "endif()\n\n" \
    "# search for libmkl_rt.so in common locations (conda env, system)\n" \
    "find_library(MKL_RT NAMES mkl_rt mkl PATHS ${{CONDA_PREFIX_DIR}}/lib /usr/lib /usr/lib64 /usr/local/lib NO_DEFAULT_PATH)\n" \
    "if(MKL_RT)\n" \
    "  message(STATUS \"Found MKL runtime: ${{MKL_RT}}\")\n" \
    "  # Link directly to the runtime library file (full path) to avoid -l name lookup issues\n" \
    "  set(MKL_LINK ${{MKL_RT}})\n" \
    "else()\n" \
    "  message(WARNING \"MKL runtime not found via find_library in CONDA_PREFIX or system paths. Falling back to link names (mkl_intel_ilp64 mkl_intel_thread mkl_core).\")\n" \
    "  set(MKL_LINK mkl_intel_ilp64 mkl_intel_thread mkl_core)\n" \
    "endif()\n" \
    "# --- end MKL detection ---\n\n" \
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
    "  $<$<COMPILE_LANGUAGE:CXX>:-pthread>\n" \
    "  $<$<COMPILE_LANGUAGE:CUDA>:-I${{Python3_INCLUDE_DIRS}}>\n" \
    ")\n\n" \
    "target_link_libraries({} PRIVATE\n" \
    "  ${{TORCH_LIBRARIES}}\n" \
    "  $<$<BOOL:${{TORCH_PYTHON_LIBRARY}}>:${{TORCH_PYTHON_LIBRARY}}>\n" \
    "  Threads::Threads\n" \
    "  ${{Python3_LIBRARIES}}\n" \
    "  ${{MKL_LINK}}\n" \
    ")\n\n" \
    "set_target_properties({} PROPERTIES\n" \
    "  CXX_STANDARD 14\n" \
    "  POSITION_INDEPENDENT_CODE ON\n" \
    ")".format(name, name, name, name, name, name, name)
    )
