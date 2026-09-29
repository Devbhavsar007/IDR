# Core C++ library
add_library(idr-core STATIC
    src/timing/timestamp_validator.cpp
)

target_include_directories(idr-core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(idr-core PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
)

# Ensure C++17
target_compile_features(idr-core PUBLIC cxx_std_17)
