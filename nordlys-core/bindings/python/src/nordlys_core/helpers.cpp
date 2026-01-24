#include "helpers.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Helper function to convert string device to ClusterBackendType
ClusterBackendType device_string_to_enum(const std::string& device) {
  if (device == "cpu" || device == "CPU") {
    return ClusterBackendType::Cpu;
  } else if (device == "cuda" || device == "CUDA") {
    return ClusterBackendType::CUDA;
  } else {
    std::string error_msg = "Invalid device: " + device + ". Must be 'cpu' or 'cuda'";
    throw nb::value_error(error_msg.c_str());
  }
}
