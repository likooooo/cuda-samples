#include "cuda_allocator.hpp"
#include <assert.h>
#include <algorithm>
#include <numeric>

void test_allocation_and_deallocation() {
    auto test = [](auto vec) { vec.resize(1024); vec.clear(); };
    test(cuda_vector_host<int>());
    test(cuda_vector_managed<int>());
    // test(cuda_vector_device<int>()); // segmentation fault
    cuda_vector_device<int>().reserve(1024);
    std::cout << "test allocation & deallocation success" << std::endl;
}

void test_memcpy() {
    cuda_vector_device<int> cuda_vec_device;
    cuda_vec_device.reserve(1024);

    std::vector<int> data(1024); std::iota(data.begin(), data.end(), 0);
    //std::cuda::iota(cuda_vec_device.data(), cuda_vec_device.c)
    cudaMemcpy(cuda_vec_device.data(), data.data(), data.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); // 同步，确保内核完成执行

    // 主机内存测试
    cuda_vector_host<int> cuda_vec_host;
    cuda_vec_host.resize(1024);
    cuda_vec_host.at(0) = 42; // 使用 at() 方法
    cuda_vec_host[0] = 42; // 直接索引访问

    // 托管内存测试
    cuda_vector_managed<int> cuda_vec_managed;
    cuda_vec_managed.resize(1024);
    cuda_vec_managed.at(0) = 42; // 使用 at() 方法
    cuda_vec_managed[0] = 42; // 直接索引访问

    std::cout << "内存分配和释放测试通过!" << std::endl;
}


int main() {
    test_allocation_and_deallocation();
    test_memcpy();
    //test_allocation_failure();
    //test_deallocation_failure();
    //test_mixed_memory_usage();

    return 0;
}