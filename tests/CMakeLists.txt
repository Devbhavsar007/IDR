# Tests
add_executable(idr-tests
    unit/test_timestamp_validator.cpp
    unit/test_geo_utils.cpp
    unit/test_quat_utils.cpp
    unit/test_strapdown_ins.cpp
    unit/test_eskf.cpp
    unit/test_alignment.cpp
    unit/test_zupt.cpp
    unit/test_map_matching.cpp
    unit/test_neural.cpp
    unit/test_edge_backend.cpp
    unit/test_jni_handle.cpp
    unit/test_osm_loader.cpp
    unit/test_replay.cpp
    unit/test_multisensor_fusion.cpp
)

target_include_directories(idr-tests PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(idr-tests PRIVATE
    idr-core
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(idr-tests)
