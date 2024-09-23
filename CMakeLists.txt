cmake_minimum_required(VERSION 3.22)
project(cuda-sample)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
else()
    message(FATAL_ERROR "CUDA not found")
endif()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_70")

include_directories(${CMAKE_CURRENT_LIST_DIR}/Common)

include_directories(${CMAKE_CURRENT_LIST_DIR}/Common/utils)
cuda_add_executable(simplePrintf ${CMAKE_CURRENT_LIST_DIR}/Samples/0_Introduction/simplePrintf/simplePrintf.cu)
target_link_libraries(simplePrintf PRIVATE ${CUBLAS_LIBRARIES})

file(GLOB list ${CMAKE_CURRENT_LIST_DIR}/Common/utils/*.cpp)
message(STATUS "add executable: ${list}")
foreach(cpp IN LISTS list)
  get_filename_component(base_name ${cpp} NAME_WLE)
  message(${base_name})
  cuda_add_executable(${base_name} ${cpp})
  target_link_libraries(${base_name} PRIVATE ${CUBLAS_LIBRARIES})
endforeach()
