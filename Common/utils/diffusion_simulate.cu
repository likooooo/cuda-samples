#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <cmath>
#include <iostream>
#include "py_helper.hpp"
#include "cuda_allocator.hpp"
#define get_thread_index(__x, __y, __z)\
    int __x = blockIdx.x* blockDim.x + threadIdx.x;\
    int __y = blockIdx.y* blockDim.y + threadIdx.y;\
    int __z = blockIdx.z* blockDim.z + threadIdx.z;
#define __both_side__ __host__ __device__ 
//== TODO : move kernel function to another head-only file
using real = float;
__both_side__ inline bool boundary_check(real x, real n, real boundary_offset = 0) {
    return (0 <= x - boundary_offset) && (x < n - boundary_offset);
};
__both_side__ inline real gauss_normalize(real sigmax, real sigmay) {
    return 1.0 / (2 * M_PI * sigmax * sigmay);
}
__both_side__ inline real gauss(const real x, const real y, const real x0, const real y0, const real sigmax, const real sigmay, const real A = 1.0) {
    return A * exp(
        -0.5 * (((x - x0) * (x - x0)/ (sigmax * sigmax))
        -0.5 * ((y - y0) * (y - y0)/ (sigmay * sigmay))) 
    );
}
//== \frac {\partial u}{\partial x} 
__both_side__ inline real derivative_1(const real u_m, const real u_p, const real delta){
    return (u_p - u_m) / delta;
}
//== \frac {\partial^2 u}{\partial x^2}
__both_side__ inline real derivative_2(const real x_m, const real x, const real x_p, const real delta){
    return (x_p + x_m - 2 * x) / (delta * delta);
}
__global__ void gauss_kernel(real* out, const real dx, const real dy, const real sigmax, const real sigmay, const int nx, const int ny, const int nz = 1){
    get_thread_index(x, y, z);
    if(!(boundary_check(x, nx) && boundary_check(y, ny))) return; 
    for(int i = 0; i < nz; i ++){
        int page_offset = i *(nx * ny);
        out[page_offset + y * nx + x] = gauss(x * dx, y * dy, nx / 2 * dx, ny / 2 * dy, sigmax, sigmay, gauss_normalize(sigmax, sigmax));
    }
}
__global__ void PDE_diffusion_equation(real* out, const real* in, const real dx, const real dy, const real dt, const real D, int nx, int ny, const int nz = 1){
    get_thread_index(x, y, z);
    if(!(boundary_check(x, nx) && boundary_check(y, ny))) return; 
    for(int i = 0; i < nz; i ++){
        int page_offset = i *(nx * ny);
        out[page_offset + y * nx + x] = in[y * nx + x] + dt * D * (
            //== \nabla^2 u
            derivative_2(in[(y - 1) * ny + x], in[y * nx + x], in[(y + 1) * ny + x], dy) + 
            derivative_2(in[y * nx + x - 1], in[y * nx + x], in[y * nx + x + 1], dx) 
        );
        in = out + page_offset; 
        __syncthreads();
    }
}

void solve_diffusion(real dt, real simulate_time, real Lx, real Ly, real dx, real dy,real sigmax, real sigmay, real D) {
    const int nx = static_cast<int>(Lx / dx) + 1;
    const int ny = static_cast<int>(Ly / dy) + 1;
    const int nz = 1;
    const int iter_count = static_cast<int>(simulate_time / dt) / nz + 1;
    const dim3 blockSize(16, 16);
    const dim3 gridSize(aligin_count(nx, blockSize.x), aligin_count(ny, blockSize.y), nz);

    cuda_vector_device<real> mem_device_in; mem_device_in.reserve(nx * ny * nz);
    cuda_vector_device<real> mem_device_out; mem_device_out.reserve(nx * ny * nz);
    cuda_vector_host<real> result(nx * ny * nz);

    real *f = mem_device_out.data(), *f0 = mem_device_in.data();
    gauss_kernel << <gridSize, blockSize >> > (f0, dx, dy, sigmax, sigmay, nx, ny, nz);
    for (int k = 0; k < iter_count; ++k) {
        PDE_diffusion_equation << <gridSize, blockSize >> > (f, f0, dx, dy, dt, D, nx, ny, nz);
        //== TODO : record current frame
        cudaMemcpy(result.data(), f, result.size() * sizeof(real), cudaMemcpyDeviceToHost);
        //== set f as f0 for next iter
        std::swap(f, f0);
    }
}

struct diffusion_input_params{
    const real D = 0.1;     // 扩散系数
    const real dx = 0.01;   // 空间步长
    const real dy = 0.01;   // 空间步长
    const real Lx = 1.0;    // 区域长度
    const real Ly = 1.0;    // 区域宽度
    const real dt = 0.001;  // 时间步长
    const real T = 1.0;     // 仿真时间
    const real sigmax = 0.1;// 高斯分布标准差
    const real sigmay = 0.1; 
};
int main() {
    const diffusion_input_params param;
    solve_diffusion(
        param.dt, param.T, 
        param.Lx, param.Ly, 
        param.dx, param.dy, 
        param.sigmax, param.sigmay,
        param.D
    );
    return 0;
}









// 
//
//#define N 10240
//#define TIMESTEPS 1000
//#define DELTA_T 0.1
//#define D 1.0 // 扩散系数
//
//__global__ void diffuseKernel(real* matrix, real* newMatrix, int n) {
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    const unsigned slope_step = 1;
//    auto check_in_grid = [n](unsigned x, unsigned step = 0){
//        return (0 <= (int)x - step) && (x <= n - step);
//    };
//    if(!(check_in_grid(x) && check_in_grid(y))) return;
//    if(!(check_in_grid(x, slope_step) && check_in_grid(y, slope_step))){
//        newMatrix[y * n + x] = 0;
//        return;
//    }
//
//    const real scalar = D * DELTA_T / (slope_step * slope_step);
//    newMatrix[y * n + x] = matrix[y * n + x] + scalar * (
//        (matrix[(y - 1) * n + x] - 2 * matrix[y * n + x] + matrix[(y + 1) * n + x]) +
//        (matrix[y * n + (x - 1)] - 2 * matrix[y * n + x] + matrix[y * n + (x + 1)])
//    );
//}
//
//void initialize_impuse(real* matrix, int n)
//{
//    std::memset(matrix, 0, n * n * sizeof(real));
//    matrix[n * n / 2 + n / 2] = 1;// n* n;
//}
//void initializeMatrix(real* matrix, int n, real sigma = 10) {
//    //initialize_impuse(matrix, n);
//    //return;
//    //sigma *= sigma;
//    sigma = n * 0.2 / 3;
//    real xCenter = n / 2.0;
//    real yCenter = n / 2.0;
//    real norm = 1.0 / (2 * M_PI * sigma);
//    for (int y = 0; y < n; y++) {
//        for (int x = 0; x < n; x++) {
//            matrix[y * n + x] = norm * exp(-((x - xCenter) * (x - xCenter) + (y - yCenter) * (y - yCenter)) / (2 * sigma));
//        }
//    }
//}
//
//// 使用Boost.Python将数据传递给Python
//void sendDataToPython(real* matrix, int n, py::object& update_func) {
//    np::ndarray data = np::from_data(matrix, np::dtype::get_builtin<real>(),
//        py::make_tuple(n, n),
//        py::make_tuple(sizeof(real) * n, sizeof(real)),
//        py::object());
//    catch_py_error(update_func(data));
//}
//
//
//int main() {
//    // Py_Initialize();
//    // np::initialize();
//
//    // // 导入Python可视化模块
//    // py::object main_module = py::import("__main__");
//    // py::object main_namespace = main_module.attr("__dict__");
//
//    // // 打印Python路径
//    // py::object sys = py::import("sys");
//    // py::object path = sys.attr("path");
//    // std::array<char, 1024> cwd;
//    // getcwd(cwd.data(), sizeof(cwd));
//    // std::cout << "Python path:" << std::endl;
//    // py::list pathList = py::extract<py::list>(path);
//    // pathList.append(PY_SOURCE_DIR);
//
//    // int listSize = len(pathList);
//    // for (int i = 0; i < listSize; ++i) {
//    //     py::object temp = pathList[i];
//    //     std::cout << (std::string)py::extract<std::string>(temp)<< std::endl;
//    // }
//    // safe_exec("import visualizer", main_namespace);
//    // py::object update_func = py::eval("visualizer.update", main_namespace);
//
//    py_loader::init();
//    py_loader loader;
//    auto pathList = loader.add_env_path();
//    int listSize = len(pathList);
//    for (int i = 0; i < listSize; ++i) {
//        py::object temp = pathList[i];
//        std::cout << (std::string)py::extract<std::string>(temp)<< std::endl;
//    }
//    //loader.exec("import visualizer");
//    //py::object update_func = loader.eval<py::object>("visualizer.update");
//
//    catch_py_error(py::exec("import visualizer", loader.py_namespace));
//    py::object update_func;
//    catch_py_error(update_func = py::eval("visualizer.update", loader.py_namespace));
//
//    // 初始化扩散矩阵
//    real* h_matrix = new real[N * N];
//    initializeMatrix(h_matrix, N);
//
//    // 分配CUDA内存
//    real* d_matrix, * d_newMatrix;
//    cudaMalloc((void**)&d_matrix, N * N * sizeof(real));
//    cudaMalloc((void**)&d_newMatrix, N * N * sizeof(real));
//
//    // 将初始矩阵拷贝到CUDA内存
//    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(real), cudaMemcpyHostToDevice);
//
//    dim3 blockSize(16, 16);
//    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
//
//    sendDataToPython(h_matrix, N, update_func);
//    for (int t = 0; t < TIMESTEPS; t++) {
//        diffuseKernel << <gridSize, blockSize >> > (d_matrix, d_newMatrix, N);
//        cudaMemcpy(d_matrix, d_newMatrix, N * N * sizeof(real), cudaMemcpyDeviceToDevice);
//
//        cudaMemcpy(h_matrix, d_matrix, N * N * sizeof(real), cudaMemcpyDeviceToHost);
//        sendDataToPython(h_matrix, N, update_func);
//    }
//    // 清理内存
//    cudaFree(d_matrix);
//    cudaFree(d_newMatrix);
//    delete[] h_matrix;
//
//    return 0;
//}
