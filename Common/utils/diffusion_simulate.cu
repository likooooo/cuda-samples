#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <cmath>
#include <map>
#include <iostream>
#include <functional>
#include "py_helper.hpp"
#include "cuda_allocator.hpp"
#define check_cuda_error() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define get_thread_index_2d(__x, __y)\
    int __x = blockIdx.x* blockDim.x + threadIdx.x;\
    int __y = blockIdx.y* blockDim.y + threadIdx.y;
#define get_thread_index_3d(__x, __y, __z)\
    int __x = blockIdx.x* blockDim.x + threadIdx.x;\
    int __y = blockIdx.y* blockDim.y + threadIdx.y;\
    int __z = blockIdx.z* blockDim.z + threadIdx.z;

#define __both_side__ __host__ __device__ 

//== TODO : move kernel function to another head-only file
using real = float;
__both_side__ inline bool boundary_check(real x, real n, real boundary_offset = 0) {
    return (0 <= x - boundary_offset) && (x < n - boundary_offset);
}
__both_side__ inline real gauss_normalize(real sigmax, real sigmay) {
    return 1.0 / (2 * M_PI * sigmax * sigmay);
}
__both_side__ inline real gauss_normalize(real sigma) {
    return gauss_normalize(sigma, sigma);
}
__both_side__ inline real gauss(const real x, const real y, const real x0, const real y0, const real sigmax, const real sigmay, const real A) {
    return A * std::exp(
        -0.5 * ((x - x0) * (x - x0)/ (sigmax * sigmax))
        -0.5 * ((y - y0) * (y - y0)/ (sigmay * sigmay))
    );
}
__both_side__ inline real gauss(const real x, const real y, const real x0, const real y0, const real sigma, const real A) {
    return gauss(x, y, x0, y0, sigma, sigma, A);
}
//== \frac {\partial u}{\partial x} 
__both_side__ inline real derivative_1(const real u_m, const real u_p, const real delta){
    return (u_p - u_m) / delta;
}
//== \frac {\partial^2 u}{\partial x^2}
__both_side__ inline real derivative_2(const real x_m, const real x, const real x_p, const real delta){
    return (x_p + x_m - 2 * x) / (delta * delta);
}
__global__ void gauss_kernel(real* out, const real dx, const real dy, const real sigmax, const real sigmay, const real A,const int nx, const int ny, const int nz = 1){
    get_thread_index_3d(x, y, z);
    if(!(boundary_check(x, nx) && boundary_check(y, ny) && boundary_check(z, nz))) return;
    for(int i = 0; i < nz; i ++){
        int page_offset = i *(nx * ny);
        out[page_offset + y * nx + x] = gauss(x * dx, y * dy, 0.5 * nx  * dx, 0.5 * ny * dy, sigmax, sigmay, A);
    }
}
__global__ void PDE_diffusion_equation(real* out, const real* in, const real dx, const real dy, const real dt, const real D, int nx, int ny, const int nz = 1){
    get_thread_index_3d(x, y, z);
    if (!(boundary_check(x, nx) && boundary_check(y, ny) && boundary_check(z, nz))) return;
    for(int i = 0; i < nz; i ++){
        int page_offset = i *(nx * ny);
        out[page_offset + y * nx + x] = in[y * nx + x] + dt * D * (
            //== \nabla^2 u
            derivative_2(in[(y - 1) * ny + x] * dy, in[y * nx + x] * dy, in[(y + 1) * ny + x] * dy, dy) +
            derivative_2(in[y * nx + x - 1] * dx, in[y * nx + x] * dx, in[y * nx + x + 1] * dx, dx)
        );
        in = out + page_offset; 
        __syncthreads();
    }
}


void solve_diffusion_with_pde(real dt, real simulate_time, real Lx, real Ly, real dx, real dy,real sigmax, real sigmay, real D, std::function<bool(np::ndarray)> callback = nullptr) {
    const int nx = std::ceil(Lx / dx);
    const int ny = std::ceil(Ly / dy);
    const int nz = 1;
    const int iter_count = static_cast<int>(simulate_time / dt) / nz + 1;
    const dim3 blockSize(16, 16);
    const dim3 gridSize(aligin_count(nx, blockSize.x), aligin_count(ny, blockSize.y), nz);

    cuda_vector_device<real> mem_device_in; mem_device_in.reserve(nx * ny * nz);
    cuda_vector_device<real> mem_device_out; mem_device_out.reserve(nx * ny * nz);
    cuda_vector_host<real> result(nx * ny * nz);

    real *f = mem_device_out.data(), *f0 = mem_device_in.data();
    gauss_kernel << <gridSize, blockSize >> > (f0, dx, dy, sigmax, sigmay, 1.0 /*gauss_normalize(sigmax * nx, sigmax * nx)*/, nx, ny, nz);
    cudaMemcpy(result.data(), f0, result.size() * sizeof(real), cudaMemcpyDeviceToHost);

    if(!callback(create_ndarray_from_vector(result, { nx, ny }))) return;
    for (int k = 0; k < iter_count; ++k) {
        PDE_diffusion_equation << <gridSize, blockSize >> > (f, f0, dx, dy, dt, D, nx, ny, nz);
        cudaDeviceSynchronize();

        cudaMemcpy(result.data(), f, result.size() * sizeof(real), cudaMemcpyDeviceToHost);
        if (!callback(create_ndarray_from_vector(result, { nx, ny }))) return;
        //== set f as f0 for next iter
        std::swap(f, f0);
    }
}

//////////////////////////////
struct sigmoid_solver{
    real scalar{1}, bais{0};
    real mapping_accuracy_to_span(const real min, const real max, const real accuracy = 1e-2) {
        //== mapping x from [-3, 3] to [0.05, 0.95]
        const std::map<real, real> mapping_accuracy_to_span{ {1e-2f, 3.0f} };
        assert(std::abs((max - min) / 2) > accuracy * 9);
        return mapping_accuracy_to_span.at(accuracy);
    }
    __both_side__ void mapping_to(const real min, const real max, const real span = 3.0f) {
        bais = (min + max) / 2;
        scalar = span / ((max - min) / 2);
    }
    __both_side__ real solve(real x){
        //== y = \frac{1}{1 + e^(scalar(-x) + bais)}
        return 1.0 / (1 + std::exp(-scalar * (x + bais)));
    }
};
__global__ void initialize_kernel(real* f, int nx, int ny){
    get_thread_index_2d(idx, idy);
    if(!(boundary_check(idx, nx) && boundary_check(idy, ny))) return; 
    auto cal_delta =[](int n)->real{return 1.0/(n - 1);};
    real dx = cal_delta(nx); 
    real dy = cal_delta(ny); 

    real x = idx * dx;
    real y = idy * dy;

    const real min_boundary = 0.4;
    const real max_boundary = 0.6;
    const real smooth_ratio = 0.75;
    sigmoid_solver left_boundary,right_boundary;
    left_boundary.mapping_to(min_boundary * smooth_ratio, min_boundary);
    right_boundary.mapping_to(max_boundary + (1.0 - max_boundary) * smooth_ratio, max_boundary);
    auto cal_f = [&](real n)->real {
        if (min_boundary < n && n < max_boundary) return 1.0;
        else if (min_boundary >= n) return left_boundary.solve(n);
        else return right_boundary.solve(n);
    };
    f[idy * nx + idx] = cal_f(x) * cal_f(y);
}
__global__ void green_function_kernel(real* g, const int nx, const int ny, real t, const real D = 1.0) {
    get_thread_index_2d(idx, idy);
    if(!(boundary_check(idx, nx) && boundary_check(idy, ny))) return; 
    auto cal_delta =[](int n)->real{return 1.0/(n - 1);};
    real x = idx * cal_delta(nx); 
    real y = idy * cal_delta(ny); 

    real sigma = std::sqrt(2.0 * D * t);
    g[idy * nx + idx] = gauss(x, y, 0.5, 0.5, sigma, gauss_normalize(sigma));
}
// 使用cuFFT进行二维快速傅里叶变换和卷积（单精度，原地变换）
void fft_convolve(float* d_f, float* d_g, float* d_result, int Nx, int Ny) {
    //cufftHandle plan;
    //cufftPlan2d(&plan, Nx, Ny, CUFFT_R2C, 1); // 实数到复数的前向变换计划

    //cufftComplex* d_fft_f = d_f;
    //cufftComplex* d_fft_g = d_g;
    //// cudaMalloc((void**)&d_fft_f, Nx * Ny * sizeof(cufftComplex));
    //// cudaMalloc((void**)&d_fft_g, Nx * Ny * sizeof(cufftComplex));

    //// Convert real to complex
    //// cudaMemcpy(d_fft_f, d_f, Nx * Ny * sizeof(float), cudaMemcpyDeviceToDevice);
    //// cudaMemcpy(d_fft_g, d_g, Nx * Ny * sizeof(float), cudaMemcpyDeviceToDevice);

    //// Perform 2D FFT in-place
    //cufftExecR2C(plan, d_f, reinterpret_cast<cufftComplex*>(d_fft_f));
    //cufftExecR2C(plan, d_g, reinterpret_cast<cufftComplex*>(d_fft_g));

    //// Multiply the results of the FFTs
    //for (int iy = 0; iy < Ny; ++iy) {
    //    for (int ix = 0; ix < Nx; ++ix) {
    //        int index = iy * Nx + ix;
    //        cufftComplex temp = {
    //            d_fft_f[index].x * d_fft_g[index].x - d_fft_f[index].y * d_fft_g[index].y,
    //            d_fft_f[index].x * d_fft_g[index].y + d_fft_f[index].y * d_fft_g[index].x
    //        };
    //        d_fft_f[index] = temp;
    //    }
    //}

    //// Create inverse plan for C2R
    //cufftDestroy(plan);
    //cufftPlan2d(&plan, Nx, Ny, CUFFT_C2R, 1); // 复数到实数的逆向变换计划

    //// Perform inverse 2D FFT in-place
    //cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(d_fft_f), d_result);

    //// Normalize result
    //float normFactor = 1.0f / (Nx * Ny);
    //for (int iy = 0; iy < Ny; ++iy) {
    //    for (int ix = 0; ix < Nx; ++ix) {
    //        int index = iy * Nx + ix;
    //        d_result[index] *= normFactor;
    //    }
    //}

    //cudaFree(d_fft_f);
    //cudaFree(d_fft_g);
    //cufftDestroy(plan);
}

int solve_diffusion_with_green_function() {
    //// 参数定义
    //const real D = 0.1; // 扩散系数
    //const int nx = 1000; // 空间点数 x 方向
    //const int ny = 1000; // 空间点数 y 方向
    //const int T = 1000; // 时间步数
    //const real dt = 0.01; // 时间步长

    //cuda_vector_host<real> solution(nx * ny);
    //cuda_vector_device<real> d_f; d_f.reserve(nx * ny);
    //cuda_vector_device<real> d_g; d_g.reserve(nx * ny);
    //real* f = d_f.data(), *g = d_g.data();

    //dim3 blockSize(16, 16);
    //dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
    //              (ny + blockSize.y - 1) / blockSize.y);
    //initialize_kernel<<<gridSize, blockSize>>>(f, nx, ny);
    //cudaDeviceSynchronize(); cudaCheckError();
    //for (int t = 1; t <= T; ++t) {
    //    // 计算格林函数
    //    green_function_kernel<<<gridSize, blockSize>>>(g, nx, ny, t);
    //    cudaDeviceSynchronize(); cudaCheckError();
    //    // 执行卷积
    //    fft_convolve(f, g, f, nx * ny);
    //    cudaMemcpy(solution.data(), f, nx * ny * sizeof(real), cudaMemcpyDeviceToHost);
    //}

    //// 输出最终状态
    //for (int iy = 0; iy < ny; ++iy) {
    //    for (int ix = 0; ix < nx; ++ix) {
    //        real x = ix / (real)(nx - 1);
    //        real y = iy / (real)(ny - 1);
    //        std::cout << x << " " << y << " " << solution[iy * nx + ix] << std::endl;
    //    }
    //}

    //// 清理资源
    //cudaFree(d_f);
    //cudaFree(d_g);
    //cudaFree(d_result);
    return 0;
}

int main() {
    py_loader::init();
    init_stl_converters<std::vector<int>>();

    struct diffusion_input_params {
        const real D = 1;     // 扩散系数
        const real dx = 0.01;   // 空间步长
        const real dy = 0.01;   // 空间步长
        const real Lx = 1.0;    // 区域长度
        const real Ly = 1.0;    // 区域宽度
        const real dt = 0.001;  // 时间步长
        const real T = 1.0;     // 仿真时间
        const real sigmax = 0.1;// 高斯分布标准差
        const real sigmay = 0.1;
    }param;
    solve_diffusion_with_pde(
        param.dt, param.T, 
        param.Lx, param.Ly, 
        param.dx, param.dy, 
        param.sigmax, param.sigmay,
        param.D, py_plot::create_callback_simulation_fram_done()
    );
    return 0;
}
