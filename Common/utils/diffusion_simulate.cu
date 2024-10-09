#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <cmath>
#include <map>
#include <iostream>
#include <functional>
#include "py_helper.hpp"
#include "kernels.hpp"

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
    cudaDeviceSynchronize(); check_cuda_error();
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
    auto cal_delta =[](int n)->real{return 1.0/ n;};
    real x = idx * cal_delta(nx); 
    real y = idy * cal_delta(ny); 

    real sigma = std::sqrt(2.0 * D * t) * nx;
    g[idy * nx + idx] = gauss(idx, idy, 0.5 * nx, 0.5 * ny, sigma, gauss_normalize(sigma));
}
void solve_diffusion_with_green_function(real dt, real simulate_time, real Lx, real Ly, real dx, real dy, real sigmax, real sigmay, real D, std::function<bool(np::ndarray)> callback) {
    const int nx = std::ceil(Lx / dx);
    const int ny = std::ceil(Ly / dy);
    const int nz = 1;
    const int iter_count = static_cast<int>(simulate_time / dt) / nz + 1;
    const dim3 blockSize(16, 16);
    const dim3 gridSize(aligin_count(nx, blockSize.x), aligin_count(ny, blockSize.y), nz);

    auto f0 = make_inplace_fft_vec<cuda_allocator<real, cuda_memory_type::device>>(nx, ny);
    auto g = make_inplace_fft_vec<cuda_allocator<real, cuda_memory_type::device>>(nx, ny);

    gauss_kernel << <gridSize, blockSize >> > (f0.data(), dx, dy, sigmax, sigmay, 1.0 /*gauss_normalize(sigmax * nx, sigmax * nx)*/, nx, ny, nz);
    cudaDeviceSynchronize(); check_cuda_error();

    cuda_vector_host<real> result; result << f0;
    if (!callback(create_ndarray_from_vector(result, { nx, ny }))) return;
    cudaDeviceSynchronize(); check_cuda_error();
    for (int k = 1; k < iter_count; ++k) {
        green_function_kernel << <gridSize, blockSize >> > (g.data(), nx, ny, dt);
        cudaDeviceSynchronize(); check_cuda_error();
        fft_convolve(f0, f0, g, nx, ny);
        cudaDeviceSynchronize(); check_cuda_error();
        result << f0;
        if (!callback(create_ndarray_from_vector(result, { nx, ny }))) return;
    }
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
    bool use_pde = false;
    if(use_pde){
        solve_diffusion_with_pde(
            param.dt, param.T, 
            param.Lx, param.Ly, 
            param.dx, param.dy, 
            param.sigmax, param.sigmay,
            param.D, py_plot::create_callback_simulation_fram_done()
        );
    }
    else{
        solve_diffusion_with_green_function(
            param.dt, param.T,
            param.Lx, param.Ly,
            param.dx, param.dy,
            param.sigmax, param.sigmay,
            param.D, py_plot::create_callback_simulation_fram_done()
        );
    }
    return 0;
}
