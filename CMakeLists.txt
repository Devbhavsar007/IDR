cmake_minimum_required(VERSION 3.20)
project(idr VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ──────────────────────────────────────────────
# Options
# ──────────────────────────────────────────────
option(IDR_BUILD_TESTS    "Build unit and integration tests" ON)
option(IDR_BUILD_EDGE     "Build edge deployment executable" OFF)
option(IDR_BUILD_REPLAY   "Build replay/simulation tools"    ON)
option(IDR_ENABLE_ASAN    "Enable AddressSanitizer"          OFF)
option(IDR_ENABLE_UBSAN   "Enable UndefinedBehaviorSanitizer" OFF)

# ──────────────────────────────────────────────
# Compiler warnings
# ──────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
    if(IDR_ENABLE_ASAN)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
    if(IDR_ENABLE_UBSAN)
        add_compile_options(-fsanitize=undefined)
        add_link_options(-fsanitize=undefined)
    endif()
elseif(MSVC)
    add_compile_options(/W4 /permissive-)
endif()

# ──────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)

# Optional dependencies (enabled as modules are built)
# find_package(GeographicLib REQUIRED)
# find_package(onnxruntime REQUIRED)
# find_package(SQLiteCpp REQUIRED)

# ──────────────────────────────────────────────
# Core library
# ──────────────────────────────────────────────
add_subdirectory(core)

# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────
if(IDR_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()

# ──────────────────────────────────────────────
# Edge executable
# ──────────────────────────────────────────────
if(IDR_BUILD_EDGE)
    add_subdirectory(edge)
endif()

# ──────────────────────────────────────────────
# Replay / simulation tools
# ──────────────────────────────────────────────
if(IDR_BUILD_REPLAY)
    add_subdirectory(simulation)
endif()

# ──────────────────────────────────────────────
# Android JNI library
# ──────────────────────────────────────────────
add_subdirectory(android)

