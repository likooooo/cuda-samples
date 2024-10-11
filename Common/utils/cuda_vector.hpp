#pragma once
#include <cublas_v2.h>
#include <cufft.h>
#include "cuda_allocator.hpp"
#include "common.h"

#define check_cuda_error() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// CUDA API error checking
#ifndef CUDA_RT_CALL
#   define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL

// cufft API error chekcing
#ifndef CUFFT_CALL
#   define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUFFT_CALL

#define get_thread_index_1d(__x)\
    int __x = blockIdx.x* blockDim.x + threadIdx.x;
#define get_thread_index_2d(__x, __y)\
    int __x = blockIdx.x* blockDim.x + threadIdx.x;\
    int __y = blockIdx.y* blockDim.y + threadIdx.y;
#define get_thread_index_3d(__x, __y, __z)\
    int __x = blockIdx.x* blockDim.x + threadIdx.x;\
    int __y = blockIdx.y* blockDim.y + threadIdx.y;\
    int __z = blockIdx.z* blockDim.z + threadIdx.z;

#define __both_side__ __host__ __device__ 

//== basic operator
#define VECTOR_SIZE_CHECK(lhs,rhs) do{if (lhs.capacity() != rhs.capacity()) RANGE_EXCEPTION((std::ostringstream() << "cuda_vector size not equal. " << lhs.size() << " != " << rhs.size()).str());}while(0)
static  cublasHandle_t cuBlasHandle() {
    struct wrapper {
        cublasHandle_t handle;
        wrapper() {
            cublasCreate(&handle);
            check_cuda_error();
        }
        ~wrapper() {
            cublasDestroy(handle);
            check_cuda_error();
        }
    };
    static wrapper w;
    return w.handle;
}

//== TODO : move kernel function to another head-only file
using real = float;
__both_side__ inline bool boundary_check(real x, real n, real boundary_offset = 0) {
    return (0 <= x - boundary_offset) && (x < n - boundary_offset);
}


template<class T> __global__  void pointwise_mul(T* out, const T* in, int N) {
    get_thread_index_1d(x); if (!boundary_check(x, N)) return;
    out[x] *= in[x];
}
template<class T, cuda_memory_type N> inline cuda_vector<T, N>& operator *=(cuda_vector<T, N>& lhs, const cuda_vector<T, N>& rhs) {
    VECTOR_SIZE_CHECK(lhs, rhs);
    dim3 threadsPerBlock(256);
    dim3 numBlocks((lhs.capacity() + threadsPerBlock.x - 1) / threadsPerBlock.x);
    pointwise_mul<T> << <numBlocks, threadsPerBlock >> > (lhs.data(), rhs.data(), lhs.capacity());
    return lhs;
}
template<class T, cuda_memory_type N> inline cuda_vector<T, N>& operator *=(cuda_vector<T, N>& lhs, const T value) {
    cublasSscal(cuBlasHandle(), lhs.capacity(), &value, lhs.data(), 1);
    return lhs;
}
template<class T, cuda_memory_type N> inline void operator +=(cuda_vector<T, N>& lhs, const cuda_vector<T, N>& rhs) {

}
template<class T, cuda_memory_type N> inline void operator +=(cuda_vector<T, N>& lhs, const T value) {

}
template<cuda_memory_type From, cuda_memory_type To> inline cudaMemcpyKind get_cuda_memcpy_forward() {
    if (From == To) {
        return (From == cuda_memory_type::device) || (From == cuda_memory_type::managed) ? cudaMemcpyDeviceToDevice : cudaMemcpyDefault;
    }
    if (From == cuda_memory_type::device && To == cuda_memory_type::pinned) {
        return cudaMemcpyDeviceToHost;
    }
    return cudaMemcpyDefault;
}
template<class T, cuda_memory_type From, cuda_memory_type To> inline void operator << (cuda_vector<T, To>& lhs, const cuda_vector<T, From>& rhs) {
    if (lhs.capacity() < rhs.capacity()) {
        cuda_vector<T, To> temp;
        temp.reserve(rhs.capacity());
        lhs.swap(temp);
    }
    cudaMemcpy(lhs.data(), rhs.data(), sizeof(T)* rhs.capacity(), get_cuda_memcpy_forward<From, To>());
}
//== deleter
struct cufft_plan_deleter {
    void operator()(cufftHandle* plan) const {
        cufftDestroy(*plan);
        delete plan;
    }
};

inline void padding_x_for_fft(real *pData,int nx, int ny) {
    int stride = (nx / 2 + 1) * 2;
    real* p = pData + (ny - 1) * nx, * p1 = pData + (ny - 1) * stride;
    for (int y = ny - 1; y >= 0; y--) {
        cudaMemcpy(p1, p, nx * sizeof(real), cudaMemcpyDeviceToDevice);
        cudaMemset(p1 + nx, 0, sizeof(real) * (stride - nx));
        p1 -= stride;
        p -= nx;
    }
}
inline void remove_padding_after_fft(real* pData, int nx, int ny) {
    int stride = (nx / 2 + 1) * 2;
    real* p = pData + nx, * p1 = pData + stride;
    for (int y = 1; y < ny; y++) {
        cudaMemcpy(p, p1, nx * sizeof(real), cudaMemcpyDeviceToDevice);
        p1 += stride;
        p += nx;
    }
    cudaMemset(p, 0, sizeof(real) * (stride - nx) * ny);
}

template <class T>
inline void CenterCornerFlip(T* image, int widht, int height){
    const int sizeX = widht;
    const int sizeY = height;
    const int halfSizeX = sizeX / 2;
    const int halfSizeY = sizeY / 2;

    T* pA = image;
    T* pB = image + halfSizeX + sizeX % 2;
    T* pC = image + (halfSizeY + sizeY % 2) * sizeX;
    T* pD = image + (halfSizeY + sizeY % 2) * sizeX + halfSizeX + sizeX % 2;

    for (int i = 0; i < halfSizeY; i++)
    {
        cublasSswap(cuBlasHandle(), halfSizeX, pA, 1, pD, 1);
        cublasSswap(cuBlasHandle(), halfSizeX, pB, 1, pC, 1);
        pA += sizeX;
        pD += sizeX;
        pB += sizeX;
        pC += sizeX;
    }
}
template<class TVec, class T = typename TVec::value_type> inline void fft_convolve(TVec& K_1, cuda_vector_device<T>& I, cuda_vector_device<T>& K, int nx, int ny) {
    auto create_plan = [&](cufftType t) {

        std::unique_ptr<cufftHandle, cufft_plan_deleter> pPlan(new cufftHandle, cufft_plan_deleter());
        CUFFT_CALL(cufftPlan2d(pPlan.get(), ny, nx, t));
        return pPlan;
    };
    auto fft = create_plan(CUFFT_R2C);
    auto ifft = create_plan(CUFFT_C2R);
    cuda_vector_device<cuComplex>& I_c = reinterpret_cast<cuda_vector_device<cuComplex>&>(I);
    cuda_vector_device<cuComplex>& K_c = reinterpret_cast<cuda_vector_device<cuComplex>&>(K);
    T normFactor = T(1.0) / (nx * ny);
    I *= normFactor;

    padding_x_for_fft(I.data(), nx, ny);
    cufftExecR2C(*fft, I.data(), (cufftComplex*)I_c.data()); //cufftExecD2Z
    padding_x_for_fft(K.data(), nx, ny);
    cufftExecR2C(*fft, K.data(), (cufftComplex*)K_c.data());
    I_c *= K_c;
    cufftExecC2R(*ifft, (cufftComplex*)I_c.data(), I.data());
    remove_padding_after_fft(I.data(), nx, ny);
    CenterCornerFlip(I.data(), nx, ny);
}
