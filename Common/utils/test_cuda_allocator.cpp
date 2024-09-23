#include "cuda_allocator.hpp"
#include <assert.h>


void test_allocation_and_deallocation() {
    auto test = [](auto vec) { vec.reserve(1024); vec.clear(); };
    test(cuda_vector_device<int>());
    test(cuda_vector_host<int>());
    test(cuda_vector_managed<int>());
    std::cout << "test allocation & deallocation success" << std::endl;
}
//
//void test_allocation_failure() {
//    try {
//        cuda_vector_device<int> cuda_vec_device();
//        //cuda_vec_device.resize(1024 * 1024 * 1024); // ��������󣬿��ܻᵼ�·���ʧ��
//        //cuda_vec_device.clear();
//        assert(false); // Ӧ���׳��쳣
//    }
//    catch (const std::bad_alloc& e) {
//        std::cout << "�ڴ����ʧ�ܲ���ͨ��!" << std::endl;
//    }
//}
//
//void test_deallocation_failure() {
//    try {
//        cuda_vector_device<int> cuda_vec_device();
//        //cuda_vec_device.resize(1024);
//        //cuda_vec_device.at(0) = 42; 
//        //cuda_vec_device.clear();
//        //cuda_vec_device.resize(1024); // ���·����ڴ�
//        //cuda_vec_device.clear();
//        assert(true); // ��������²�Ӧ���׳��쳣
//    }
//    catch (const std::runtime_error& e) {
//        std::cout << "�ڴ��ͷ�ʧ�ܲ���ͨ��!" << std::endl;
//    }
//}
//
//void test_mixed_memory_usage() {
//    cuda_vector_device<int> cuda_vec_device();
//    cuda_vector_host<int> cuda_vec_host();
//    cuda_vector_managed<int> cuda_vec_managed();
//
//    //cuda_vec_device.resize(1024);
//    //cuda_vec_host.resize(1024);
//    //cuda_vec_managed.resize(1024);
//
//    //cuda_vec_device.at(0) = 42; 
//    //cuda_vec_host.at(0) = 42; 
//    //cuda_vec_managed.at(0) = 42; 
//
//    //cuda_vec_device.clear();
//    //cuda_vec_host.clear();
//    //cuda_vec_managed.clear();
//
//    assert(cuda_vec_device.empty());
//    assert(cuda_vec_host.empty());
//    assert(cuda_vec_managed.empty());
//
//    std::cout << "����ڴ�ʹ�ò���ͨ��!" << std::endl;
//}
//
int main() {
    test_allocation_and_deallocation();
    //test_allocation_failure();
    //test_deallocation_failure();
    //test_mixed_memory_usage();

    return 0;
}