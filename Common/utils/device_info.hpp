#pragma once
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <memory>
#include <iostream>
#include <iomanip>
#include <vector>
#include <type_traits>
inline cudaDeviceProp get_device_properties(int deviceId = 0)
{
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));
    return deviceProp;
}
inline const char* get_arch_name(const cudaDeviceProp& deviceProp)
{
    return _ConvertSMVer2ArchName(deviceProp.major, deviceProp.minor);
}
namespace details
{
    template<typename T>
    inline void printDeviceProperty(const std::string& name, T value, const std::string& unit = "") {
        std::cout << std::left << std::setw(30) << name << ": ";
        if constexpr (std::is_same<T, bool>::value) {
            std::cout << (value ? "Yes" : "No");
        }
        else if constexpr (std::is_same<T, std::string>::value) {
            std::cout << value;
        }
        else if constexpr (std::is_arithmetic_v<T>) {
            std::cout << std::fixed << std::setprecision(2) << value;
            if (!unit.empty()) {
                std::cout << " " << unit;
            }
        }
        std::cout << std::endl;
    }

}
inline void print_properties(const cudaDeviceProp& deviceProp)
{
    using namespace details;
    printDeviceProperty<std::string>("Device Name", deviceProp.name);
    printDeviceProperty<std::string>("Arch Name", get_arch_name(deviceProp));
    printDeviceProperty("Major Compute Capability", deviceProp.major);
    printDeviceProperty("Minor Compute Capability", deviceProp.minor);
    printDeviceProperty("Max Threads Per Block", deviceProp.maxThreadsPerBlock);
    printDeviceProperty("Max Threads Dim X/Y/Z", std::to_string(deviceProp.maxThreadsDim[0]) + "x" + std::to_string(deviceProp.maxThreadsDim[1]) + "x" + std::to_string(deviceProp.maxThreadsDim[2]));
    printDeviceProperty("Max Grid Size X/Y/Z", std::to_string(deviceProp.maxGridSize[0]) + "x" + std::to_string(deviceProp.maxGridSize[1]) + "x" + std::to_string(deviceProp.maxGridSize[2]));
    printDeviceProperty("Total Global Memory", static_cast<double>(deviceProp.totalGlobalMem) / (1024 * 1024), "MB");
    printDeviceProperty("Total Constant Memory", static_cast<double>(deviceProp.totalConstMem) / (1024 * 1024), "MB");
    printDeviceProperty("Shared Memory Per Block", static_cast<double>(deviceProp.sharedMemPerBlock) / (1024 * 1024), "MB");
    printDeviceProperty("Reg File Size", deviceProp.regsPerBlock);
    printDeviceProperty("Warp Size", deviceProp.warpSize);
    printDeviceProperty("Memory Pitch", static_cast<double>(deviceProp.memPitch) / (1024 * 1024), "MB");
    printDeviceProperty("Max Registers Per Block", deviceProp.regsPerBlock);
    printDeviceProperty("Clock Rate", static_cast<double>(deviceProp.clockRate) / 1e6, "GHz");
    printDeviceProperty("Total Constant Memory", static_cast<double>(deviceProp.totalConstMem) / (1024 * 1024), "MB");
    printDeviceProperty("Texture Alignment", static_cast<double>(deviceProp.textureAlignment) / (1024 * 1024), "MB");
    printDeviceProperty("Multi-Processor Count", deviceProp.multiProcessorCount);
    printDeviceProperty("Kernel Exec Timeout Enabled", deviceProp.kernelExecTimeoutEnabled);
    printDeviceProperty("Integrated", deviceProp.integrated);
    printDeviceProperty("Can Map Host Memory", deviceProp.canMapHostMemory);
    printDeviceProperty("Compute Mode", deviceProp.computeMode);
    printDeviceProperty("Max Texture 1D Linear", deviceProp.maxTexture1DLinear);
    printDeviceProperty("Async Engine Count", deviceProp.asyncEngineCount);
    printDeviceProperty("Unified Addressing", deviceProp.unifiedAddressing);
    printDeviceProperty("Memory Clock Rate", static_cast<double>(deviceProp.memoryClockRate) / 1e6, "GHz");
    printDeviceProperty("Memory Bus Width", deviceProp.memoryBusWidth, "bits");
    printDeviceProperty("L2 Cache Size", static_cast<double>(deviceProp.l2CacheSize) / (1024 * 1024), "MB");
    printDeviceProperty("Max Threads Per Multi-Processor", deviceProp.maxThreadsPerMultiProcessor);
    printDeviceProperty("Stream Priorities Supported", deviceProp.streamPrioritiesSupported);
    printDeviceProperty("Global L1 Cache Supported", deviceProp.globalL1CacheSupported);
    printDeviceProperty("Local L1 Cache Supported", deviceProp.localL1CacheSupported);
    //printDeviceProperty("Max Shared Memory Per Multi-Processor", static_cast<double>(deviceProp.maxSharedMemoryPerMultiProcessor) / (1024 * 1024), "MB");
    //printDeviceProperty("Managed Memory Supported", deviceProp.managedMemSupported);
    printDeviceProperty("Is Multi-GPU Board", deviceProp.isMultiGpuBoard);
    printDeviceProperty("Multi-GPU Board Group ID", deviceProp.multiGpuBoardGroupID);
    printDeviceProperty("Single to Double Precision Perf Ratio", deviceProp.singleToDoublePrecisionPerfRatio);
    printDeviceProperty("Pageable Memory Access", deviceProp.pageableMemoryAccess);
    printDeviceProperty("Concurrent Kernels", deviceProp.concurrentKernels);
    printDeviceProperty("PCI Domain ID", deviceProp.pciDomainID);
    printDeviceProperty("PCI Bus ID", deviceProp.pciBusID);
    printDeviceProperty("PCI Device ID", deviceProp.pciDeviceID);
    //printDeviceProperty("PCI Device Ordinal", deviceProp.pciDeviceOrdinal);
}
