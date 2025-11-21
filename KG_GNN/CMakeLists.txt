cmake_minimum_required(VERSION 3.14.0)
project(KGGNN LANGUAGES CXX CUDA)

# Allow overriding CUDA arch (eg: -DCUDA_ARCH=61)
option(CUDA_ARCH "CUDA compute capability (eg: 61 for sm_61). Leave empty for fallback 61." "")

# Find dependencies
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(Threads REQUIRED)

# --- CUDA flags: produce both sm_61 (your GTX1060)---
set(DEFAULT_CUDA_FLAGS_LIST
  -gencode=arch=compute_61,code=sm_61
  --expt-relaxed-constexpr
)

# Source files
set(SRC_DIR ${PROJECT_SOURCE_DIR}/core/src)
set(SRC_FILE
  ${SRC_DIR}/KG_GNN.cpp
  ${SRC_DIR}/gnn_run.cpp
  ${SRC_DIR}/gnn_analysis.cpp
  ${SRC_DIR}/aggregate.cu
  ${SRC_DIR}/aggregate_sddmm.cu
  ${SRC_DIR}/aggregate_gat.cu
  ${SRC_DIR}/flash_partition.cu
  ${SRC_DIR}/aggregate_gin.cu
  ${SRC_DIR}/preprocessing.cu
  ${SRC_DIR}/bin_pack.cu
)

message(STATUS "Using Python3 include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Using Python3 library: ${Python3_LIBRARIES}")
message(STATUS "Torch prefix: ${TORCH_INSTALL_PREFIX}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

# try to find torch_python library (optional)
find_library(TORCH_PYTHON_LIBRARY NAMES torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" "${Torch_DIR}/../lib" NO_DEFAULT_PATH)
if(TORCH_PYTHON_LIBRARY)
  message(STATUS "Found TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
else()
  message(WARNING "TORCH_PYTHON_LIBRARY not found automatically. If you get link errors, set TORCH_PYTHON_LIBRARY to the path of torch_python.")
endif()

# --- MKL runtime detection: prefer conda env's libmkl_rt.so if present ---
if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX_DIR $ENV{CONDA_PREFIX})
else()
  set(CONDA_PREFIX_DIR "")
endif()

if(CONDA_PREFIX_DIR)
  message(STATUS "Adding CONDA lib dir to link search: ${CONDA_PREFIX_DIR}/lib")
  link_directories("${CONDA_PREFIX_DIR}/lib")
endif()

# search for libmkl_rt.so in common locations (conda env, system)
find_library(MKL_RT NAMES mkl_rt mkl PATHS ${CONDA_PREFIX_DIR}/lib /usr/lib /usr/lib64 /usr/local/lib NO_DEFAULT_PATH)
if(MKL_RT)
  message(STATUS "Found MKL runtime: ${MKL_RT}")
  # Link directly to the runtime library file (full path) to avoid -l name lookup issues
  set(MKL_LINK ${MKL_RT})
else()
  message(WARNING "MKL runtime not found via find_library in CONDA_PREFIX or system paths. Falling back to link names (mkl_intel_ilp64 mkl_intel_thread mkl_core).")
  set(MKL_LINK mkl_intel_ilp64 mkl_intel_thread mkl_core)
endif()
# --- end MKL detection ---

# Create shared library target
add_library(KGGNN SHARED ${SRC_FILE})

# Target include dirs (ensure both C++ and CUDA see Python & Torch headers)
target_include_directories(KGGNN PRIVATE
  ${PROJECT_SOURCE_DIR}/core/include
  ${Python3_INCLUDE_DIRS}
  $<$<BOOL:${TORCH_INCLUDE_DIRS}>:${TORCH_INCLUDE_DIRS}>
)

# Ensure C++ compilation sees Python headers and pthread flag
target_compile_options(KGGNN PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-I${Python3_INCLUDE_DIRS}>
  $<$<COMPILE_LANGUAGE:CXX>:-pthread>
)

# Ensure CUDA compilation sees Python headers
target_compile_options(KGGNN PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-I${Python3_INCLUDE_DIRS}>
)

# Pass each CUDA flag separately to nvcc
foreach(_flag IN LISTS DEFAULT_CUDA_FLAGS_LIST)
  target_compile_options(KGGNN PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${_flag}>)
endforeach()

# Link libraries: Torch libs, optional torch_python, pthreads, Python libs, and MKL (resolved above)
target_link_libraries(KGGNN PRIVATE
  ${TORCH_LIBRARIES}
  $<$<BOOL:${TORCH_PYTHON_LIBRARY}>:${TORCH_PYTHON_LIBRARY}>
  Threads::Threads
  ${Python3_LIBRARIES}
  ${MKL_LINK}
)

# Properties
set_target_properties(KGGNN PROPERTIES
  CXX_STANDARD 14
  POSITION_INDEPENDENT_CODE ON
)

# Installation hints (optional)
install(TARGETS KGGNN LIBRARY DESTINATION lib)
