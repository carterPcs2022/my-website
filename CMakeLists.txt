cmake_minimum_required(VERSION 3.16)
project(zane_digital_mind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(zane_analytics_core STATIC
    cpp/src/analytics.cpp
)
target_include_directories(zane_analytics_core PUBLIC cpp/include)
target_link_libraries(zane_analytics_core PUBLIC Threads::Threads)
target_compile_options(zane_analytics_core PRIVATE -Wall -Wextra)

# Building the Python extension requires pybind11 to be discoverable
# (e.g. `pip install pybind11` and pass -Dpybind11_DIR=$(python -m pybind11 --cmakedir)).
# This target is optional: plain C++ consumers only need zane_analytics_core above.
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(zane_cpp cpp/bindings/bindings.cpp)
    target_link_libraries(zane_cpp PRIVATE zane_analytics_core)
    target_include_directories(zane_cpp PRIVATE cpp/include)
else()
    message(STATUS "pybind11 not found via CMake config; skip 'cmake --build' for the "
                    "Python module and use `pip install .` (setup.py) instead.")
endif()

enable_testing()
if(EXISTS "${CMAKE_SOURCE_DIR}/cpp/tests/test_analytics.cpp")
    add_executable(zane_analytics_tests cpp/tests/test_analytics.cpp)
    target_link_libraries(zane_analytics_tests PRIVATE zane_analytics_core)
    add_test(NAME zane_analytics_tests COMMAND zane_analytics_tests)
endif()
