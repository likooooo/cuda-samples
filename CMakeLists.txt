cmake_minimum_required(VERSION 3.22)
project(cuda-sample)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD 20)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
else()
    message(FATAL_ERROR "CUDA not found")
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_70")
cuda_add_executable(simplePrintf ${CMAKE_CURRENT_LIST_DIR}/Samples/0_Introduction/simplePrintf/simplePrintf.cu)
#target_compile_features(simplePrintf PRIVATE cxx_std_20)
target_link_libraries(simplePrintf PRIVATE ${CUBLAS_LIBRARIES})
set_target_properties(simplePrintf PROPERTIES LINKER_LANGUAGE CXX)
