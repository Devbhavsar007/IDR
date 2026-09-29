# Simulation / replay tools
add_executable(idr-replay
    replay/main.cpp
)

target_link_libraries(idr-replay PRIVATE
    idr-core
)

target_compile_features(idr-replay PUBLIC cxx_std_17)
