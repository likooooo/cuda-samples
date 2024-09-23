#include "device_info.hpp"

int main(int argc, char** argv) {
    int device_count = 0;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    for (int n = 0; n < device_count; n++) {
        std::cout << "device-" << n <<": \n";
        print_properties(get_device_properties(n));
        std::cout << "\n\n";
    }
}
