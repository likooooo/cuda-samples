#include "kernels.hpp"
#include "py_helper.hpp"
//__global__ void addForces_k(float2 *v, int nx, int ny, float fx, float fy, float effect_len) {
//    get_thread_index_2d(ix, iy);
//    if(!(boundary_check(ix, nx, r) && boundary_check(iy, ny, r))) return;
//    real tx = ix - r;
//    real ty = iy - r;
//    float s = 1.f / (1.f + tx * tx * tx * tx + ty * ty * ty * ty);
//    float2& vterm = v[nx * iy + ix];
//    vterm.x += s * fx;
//    vterm.y += s * fy;
//}

int main() {
    py_loader::init();
    int nx = 100, ny = 100;
    cuda_vector_host<real> vec(nx * ny);
    auto callback = py_plot::create_callback_simulation_fram_done(py::object(overload_click));
    while (callback(create_ndarray_from_vector(vec, { nx, ny })));
}