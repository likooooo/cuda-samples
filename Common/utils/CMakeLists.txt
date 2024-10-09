
find_package(PythonLibs 3 REQUIRED)
message(STATUS "Python libraries found: ${PYTHON_LIBRARIES}")

find_package(Boost COMPONENTS python numpy REQUIRED)
if(Boost_FOUND)
    message(STATUS "Found Boost version ${Boost_VERSION}")
    include_directories(${Boost_INCLUDE_DIRS})
    # Check if Python component of Boost is found
    if(TARGET Boost::python)
        message(STATUS "Boost.Python found")
    else()
        message(FATAL_ERROR "Boost.Python not found")
    endif()
else()
    message(FATAL_ERROR "Boost not found")
endif()

include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})


set(CMAKE_CXX_STANDARD 20)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION} at ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message(FATAL_ERROR "CUDA not found")
endif()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_70")
include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_executable(diffusion_simulate diffusion_simulate.cu)

# patch for cufft cublas
target_link_libraries(diffusion_simulate ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}  ${CUBLAS_LIBRARIES}  ${CUFFT_LIBRARIES} 
    # TODO : fix cmake
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft.so
)

# 获取当前源文件目录
set(PY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# 输出当前源文件目录
message(STATUS "Source Directory: ${PY_SOURCE_DIR}")

# 设置宏定义
add_definitions(-DPY_SOURCE_DIR="${PY_SOURCE_DIR}")