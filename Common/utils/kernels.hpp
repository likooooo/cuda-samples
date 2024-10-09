#pragma once
#include "cuda_vector.hpp"

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