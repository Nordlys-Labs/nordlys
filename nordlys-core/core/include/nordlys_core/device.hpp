#pragma once

#include <variant>

struct CpuDevice {};

struct CudaDevice {
    int id = 0;
    explicit constexpr CudaDevice(int device_id = 0) : id(device_id) {}
};

using Device = std::variant<CpuDevice, CudaDevice>;

// Helper for std::visit
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
