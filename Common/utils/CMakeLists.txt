
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

cuda_add_executable(fluid_simulate fluid_simulate.cu)
target_link_libraries(fluid_simulate ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}  ${CUBLAS_LIBRARIES}  ${CUFFT_LIBRARIES} 
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft.so
)
# 获取当前源文件目录
set(PY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# 输出当前源文件目录
message(STATUS "Source Directory: ${PY_SOURCE_DIR}")

# 设置宏定义
add_definitions(-DPY_SOURCE_DIR="${PY_SOURCE_DIR}")

##################
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(MyOpenGLProject VERSION 1.0)

# 查找 OpenGL 库
find_package(OpenGL REQUIRED)

# 如果找不到 OpenGL 库，则显示错误信息
if(NOT OpenGL_FOUND)
  message(FATAL_ERROR "OpenGL library not found.")
endif()

# 查找 GL 库的具体位置
find_library(GL_LIBRARY NAMES GL)
find_library(GLU_LIBRARY NAMES GLU)
find_package(GLUT REQUIRED)

# 输出库的位置以确认
message(STATUS "Found GL library: ${GL_LIBRARY}")
message(STATUS "Found GLU library: ${GLU_LIBRARY}")
message(STATUS "Found GLUT library: ${GLUT_LIBRARIES}")


# 添加可执行文件
set(FLUIDS_PATH ${ROOT_DIR}/Samples/5_Domain_Specific/fluidsGL)

#cuda_add_executable(my_fluids_gl ${FLUIDS_PATH}/test.cu)
cuda_add_executable(my_fluids_gl ${FLUIDS_PATH}/fluidsGL.cpp ${FLUIDS_PATH}/fluidsGL_kernels.cu)

# 链接 OpenGL 库
target_link_libraries(my_fluids_gl ${OpenGL_LIBRARIES} ${GL_LIBRARY} ${GLU_LIBRARY} ${GLUT_LIBRARIES} 
    ${CUBLAS_LIBRARIES}  ${CUFFT_LIBRARIES} 
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft.so
)

# 设置运行时路径，确保可以在运行时找到所需的库文件
# set_property(TARGET MyOpenGLApp PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
# install(DIRECTORY bin DESTINATION .)