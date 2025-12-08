from conan import ConanFile
from conan.tools.cmake import cmake_layout


class AdaptiveRouterCoreConan(ConanFile):
    name = "adaptive_router_core"
    version = "0.1.0"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        # Latest versions
        self.requires("eigen/5.0.0")
        self.requires("nlohmann_json/3.12.0")
        self.requires("msgpack-cxx/7.0.0")
        self.requires("tsl-robin-map/1.4.0")

    def build_requirements(self):
        self.test_requires("gtest/1.17.0")

    def layout(self):
        cmake_layout(self)

    def configure(self):
        # Set C++20 standard explicitly to ensure gtest and other dependencies
        # are built with compatible standards across all platforms
        self.settings.compiler.cppstd = "20"

    def validate(self):
        # Validate compiler supports C++20
        from conan.tools.build import check_min_cppstd

        check_min_cppstd(self, "20")
