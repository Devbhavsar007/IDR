# Android JNI Native Library
find_package(JNI QUIET)

if(JNI_FOUND)
    message(STATUS "JNI found: ${JNI_INCLUDE_DIRS}")
    add_library(idr-native SHARED
        jni/idr_jni.cpp
    )

    target_include_directories(idr-native PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${JNI_INCLUDE_DIRS}
        jni
    )

    target_link_libraries(idr-native PRIVATE
        idr-core
    )

    target_compile_features(idr-native PUBLIC cxx_std_17)
else()
    message(STATUS "JNI not found; skipping idr-native shared library")
endif()
