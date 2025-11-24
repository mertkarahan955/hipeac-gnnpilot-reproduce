cmake_minimum_required(VERSION 3.14.0)

project(gat LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(Threads REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" "${Torch_DIR}/../lib" NO_DEFAULT_PATH)
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

# Allow overriding CUDA arch (eg: -DCUDA_ARCH=86)
if(NOT DEFINED CUDA_ARCH)
  set(CUDA_ARCH "86")
endif()
# Enable CUDA separable compilation for device linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -rdc=true -gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
set(CMAKE_CUDA_LINK_FLAGS "${CMAKE_CUDA_LINK_FLAGS} -lcudadevrt")

include_directories(preprocessing_src)
set(SRC_DIR ${PROJECT_SOURCE_DIR})
set(SRC_FILE ${SRC_DIR}/gat.cu
${SRC_DIR}/preprocessing_src/preprocessing.cu)
include_directories(${Python3_INCLUDE_DIRS})

add_library(gat SHARED ${SRC_FILE})

target_include_directories(gat PRIVATE
  ${Python3_INCLUDE_DIRS}
  $<$<BOOL:${TORCH_INCLUDE_DIRS}>:${TORCH_INCLUDE_DIRS}>
)

target_compile_options(gat PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-I${Python3_INCLUDE_DIRS}>
  $<$<COMPILE_LANGUAGE:CXX>:-pthread>
  $<$<COMPILE_LANGUAGE:CUDA>:-I${Python3_INCLUDE_DIRS}>
)

target_link_libraries(gat PRIVATE
  ${TORCH_LIBRARIES}
  $<$<BOOL:${TORCH_PYTHON_LIBRARY}>:${TORCH_PYTHON_LIBRARY}>
  Threads::Threads
  ${Python3_LIBRARIES}
  ${MKL_LINK}
)

set_target_properties(gat PROPERTIES
  CXX_STANDARD 14
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)