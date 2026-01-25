#include "helpers.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Helper function to convert string device to Device variant
Device device_string_to_device(const std::string& device) {
  if (device == "cpu" || device == "CPU") {
    return Device{CpuDevice{}};
  } else if (device == "cuda" || device == "CUDA") {
    return Device{CudaDevice{0}};
  } else {
    std::string error_msg = "Invalid device: " + device + ". Must be 'cpu' or 'cuda'";
    throw nb::value_error(error_msg.c_str());
  }
}
