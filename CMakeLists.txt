cmake_minimum_required(VERSION 3.14.0)

project(gat LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" "${Torch_DIR}/../lib" NO_DEFAULT_PATH)
if(NOT TORCH_PYTHON_LIBRARY)
  message(WARNING "TORCH_PYTHON_LIBRARY not found. Set it manually if needed.")
endif()

# Allow overriding CUDA arch (eg: -DCUDA_ARCH=61)
if(NOT DEFINED CUDA_ARCH)
  set(CUDA_ARCH "61")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -rdc=true -gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} -lcudadevrt")

set(SRC_DIR ${PROJECT_SOURCE_DIR})
set(PREPROCESSING_DIR ${SRC_DIR}/preprocessing_src)
include_directories(${PREPROCESSING_DIR})
include_directories(${Python3_INCLUDE_DIRS})

set(SRC_FILE ${SRC_DIR}/gat.cu
${PREPROCESSING_DIR}/preprocessing.cu)

add_library(gat SHARED ${SRC_FILE})

target_include_directories(gat PRIVATE
  ${PREPROCESSING_DIR}
  ${Python3_INCLUDE_DIRS}
  $<$<BOOL:${TORCH_INCLUDE_DIRS}>:${TORCH_INCLUDE_DIRS}>
)

target_compile_options(gat PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-I${Python3_INCLUDE_DIRS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-I${Python3_INCLUDE_DIRS}>
)

target_link_libraries(gat PRIVATE
  ${TORCH_LIBRARIES}
  $<$<BOOL:${TORCH_PYTHON_LIBRARY}>:${TORCH_PYTHON_LIBRARY}>
  ${Python3_LIBRARIES}
)

set_target_properties(gat PROPERTIES
  CXX_STANDARD 14
  POSITION_INDEPENDENT_CODE ON
)