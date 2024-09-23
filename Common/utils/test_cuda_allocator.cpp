#include "cuda_allocator.hpp"
#include <assert.h>

void test_allocation_and_deallocation() {
    auto test = [](auto vec) { vec.reserve(1024); vec.clear(); };
    test(cuda_vector_device<int>());
    test(cuda_vector_host<int>());
    test(cuda_vector_managed<int>());
    std::cout << "test allocation & deallocation success" << std::endl;
}
int main() {
    test_allocation_and_deallocation();
    //test_allocation_failure();
    //test_deallocation_failure();
    //test_mixed_memory_usage();

    return 0;
}