cmake_minimum_required(VERSION 3.10)

# ��Ŀ��Ϣ
project(DiffusionSimulation VERSION 1.0)

# ����Python��
find_package(PythonLibs 3 REQUIRED)
message(STATUS "Python libraries found: ${PYTHON_LIBRARIES}")

# ����Boost��
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
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
cuda_add_executable(cuda_simulation diffusion_simulate.cu)
target_link_libraries(cuda_simulation ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${CUDA_LIBRARIES})

# 获取当前源文件目录
set(PY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# 输出当前源文件目录
message(STATUS "Source Directory: ${PY_SOURCE_DIR}")

# 设置宏定义
add_definitions(-DPY_SOURCE_DIR="${PY_SOURCE_DIR}")