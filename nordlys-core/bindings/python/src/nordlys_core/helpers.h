#pragma once

#include <nordlys_core/nordlys.hpp>
#include <string>

// Helper function to convert string device to ClusterBackendType
ClusterBackendType device_string_to_enum(const std::string& device);
