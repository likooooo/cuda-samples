#pragma once
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <iostream>
#include <vector>
#include <type_traits>
#include <string>
#include <numeric>
// 1D
// pinned -> device   : cudaMemcpyAsync : cudaMemcpyHostToDevice
// device -> pinned   : cudaMemcpyAsync : cudaMemcpyDeviceToHost //cudaDeviceSynchronize
// host_pageable -> device : cudaMemcpy : cudaMemcpyHostToDevice
// device -> host_pageable : cudaMemcpy : cudaMemcpyDeviceToHost 
// device -> device        : cudaMemcpy : cudaMemcpyDeviceToDevice
// host -> host            : std::memcpy // cudaDeviceSynchronize

// 2D
// cudaMallocPitch, cudaMemcpy2D
// cudaMallocArray, cudaArraySetComponent, cudaCreateChannelDesc, cudaGetArrayChannelDesc

//TODO :
// 1. event
// 2. resource

enum class cuda_memory_type {
    device, 
    pinned,
    pageable, 
    managed,
};
template<typename T, cuda_memory_type memory_type = cuda_memory_type::managed>
class cuda_allocator : public std::allocator<T> {
public:
    static_assert(memory_type == cuda_memory_type::pageable || std::is_trivially_copyable_v<T>);
    using base_type = std::allocator<T>;
    typedef T* pointer;

    //== std::allocator_traits
    typedef T        value_type;
    typedef size_t     size_type;
    typedef ptrdiff_t  difference_type;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal
        _GLIBCXX20_DEPRECATED_SUGGEST("std::allocator_traits::is_always_equal")
        = std::true_type;
    _GLIBCXX20_CONSTEXPR cuda_allocator() _GLIBCXX_NOTHROW { }
    template<class T1> _GLIBCXX20_CONSTEXPR cuda_allocator(const cuda_allocator<T1>& a):std::allocator<T1>(a)  { }
    cuda_allocator& operator=(const cuda_allocator&) = default;
    ~cuda_allocator() _GLIBCXX_NOTHROW { }
    template<typename U> struct rebind {
        using other = cuda_allocator<U, memory_type>;
    };
    
    pointer allocate(std::size_t n, const void* = nullptr) {
        std::size_t size = n * sizeof(T);
        void* ptr = nullptr;
        cudaError_t err = cudaSuccess;
        if constexpr (memory_type == cuda_memory_type::device) {
            err = cudaMalloc(&ptr, size);
        }
        else if constexpr (memory_type == cuda_memory_type::pinned) {
            err = cudaMallocHost(&ptr, size);
        }
        else if constexpr (memory_type == cuda_memory_type::managed) {
            err = cudaMallocManaged(&ptr, size);
        }
        else if constexpr (memory_type == cuda_memory_type::pageable){
            ptr = base_type::allocate(n);
        }
        if (err != cudaSuccess) {
            std::cerr << "cuda allocate error." << std::string(cudaGetErrorString(err)) << std::endl;
            throw std::bad_alloc();
        }
        return static_cast<pointer>(ptr);
    }
    void deallocate(pointer p, std::size_t n) {
        if (p == nullptr) return;
        cudaError_t err = cudaSuccess;
        if constexpr (memory_type == cuda_memory_type::device) {
            err = cudaFree(p);
        }
        else if constexpr (memory_type == cuda_memory_type::pinned) {
            err = cudaFreeHost(p);
        }
        else if constexpr (memory_type == cuda_memory_type::managed) {
            err = cudaFree(p);
        }
        else if constexpr (memory_type == cuda_memory_type::pageable){
            base_type::deallocate(p, n);
        }
        if (err != cudaSuccess) {
            std::cerr << "cuda deallocate error. " << std::string(cudaGetErrorString(err)) << std::endl;
            throw std::runtime_error("cuda dellocate error");
        }
    }
};
template<class T, cuda_memory_type N>  using cuda_vector = std::vector<T, cuda_allocator<T, N>>;
template<class T> using cuda_vector_device = std::vector<T, cuda_allocator<T, cuda_memory_type::device>>;
template<class T> using cuda_vector_host = std::vector<T, cuda_allocator<T, cuda_memory_type::pinned>>;
template<class T> using cuda_vector_managed = std::vector<T, cuda_allocator<T, cuda_memory_type::managed>>;
template<class T> using cuda_vector_pageable = std::vector<T>;

int cal_grid_size(int numElements,int threadsPerBlock = 1024)
{
    return (numElements + threadsPerBlock - 1) / threadsPerBlock;
}
int aligin_count(int len, int aligin){
    return (len + aligin - 1) / aligin;
}

// TODO :  make_inplace_fft_vec, make_vec(fftw_allocator.hpp)
template<class TAlloc, class TDim, class ...TDims> auto make_inplace_fft_vec(TDim d0, TDims ...rest)
{
    using T = typename TAlloc::value_type;
    using vec = std::vector<T, TAlloc>;
    std::array<TDim, sizeof...(rest) + 1> n{ d0, rest... };
    auto prod = std::accumulate(n.begin(), n.end() - 1, (size_t)1, [](auto a, auto b) {return a * b; });
    auto withpadding = (std::is_floating_point_v<T> ? (n.back() / 2 + 1) * 2 : n.back());
    vec v; v.reserve(prod * withpadding);
    return v;
}