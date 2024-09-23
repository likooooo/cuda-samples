#pragma once
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <iostream>
#include <vector>

enum class cuda_memory_type {
    device,
    host,
    managed
};
template<typename T, cuda_memory_type memory_type = cuda_memory_type::managed, std::size_t minimal_size = 1024/* bytes */>
class cuda_allocator : public std::allocator<T> {
public:
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
        using other = cuda_allocator<U, memory_type, minimal_size>;
    };
    
    pointer allocate(std::size_t n, const void* = nullptr) {
        std::size_t size = n * sizeof(T);
        void* ptr = nullptr;
        if (size >= minimal_size)
        {
            cudaError_t err = cudaSuccess;
            if constexpr (memory_type == cuda_memory_type::device) {
                err = cudaMalloc(&ptr, size);
            }
            else if constexpr (memory_type == cuda_memory_type::host) {
                err = cudaMallocHost(&ptr, size);
            }
            else if constexpr (memory_type == cuda_memory_type::managed) {
                err = cudaMallocManaged(&ptr, size);
            }
            if (err != cudaSuccess) {
                std::cerr << "cuda allocate error." << std::string(cudaGetErrorString(err)) << std::endl;
                throw std::bad_alloc();
            }
            return static_cast<pointer>(ptr);
        }
        ptr = base_type::allocate(n);
        return static_cast<pointer>(ptr);
    }
    void deallocate(pointer p, std::size_t n) {
        if (p == nullptr) return;
        if (n * sizeof(T) >= minimal_size)
        {
            cudaError_t err = cudaSuccess;
            if constexpr (memory_type == cuda_memory_type::device) {
                err = cudaFree(p);
            }
            else if constexpr (memory_type == cuda_memory_type::host) {
                err = cudaFreeHost(p);
            }
            else if constexpr (memory_type == cuda_memory_type::managed) {
                err = cudaFree(p);
            }
            if (err != cudaSuccess) {
                std::cerr << "cuda deallocate error. " << std::string(cudaGetErrorString(err)) << std::endl;
                throw std::runtime_error("cuda dellocate error");
            }
            return;
        }
        base_type::deallocate(p, n);
    }
};
template<class T> using cuda_vector_device = std::vector<T, cuda_allocator<T, cuda_memory_type::device, 1024>>;
template<class T> using cuda_vector_host = std::vector<T, cuda_allocator<T, cuda_memory_type::host, 1024>>;
template<class T> using cuda_vector_managed = std::vector<T, cuda_allocator<T, cuda_memory_type::managed, 1024>>;

