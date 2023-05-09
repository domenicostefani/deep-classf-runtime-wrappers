cmake_minimum_required(VERSION 3.16)
project(TFliteWrapper C CXX)
set(WRAP_LIB_NAME tflitewrapper)

include(FetchContent)

FetchContent_Declare(
    tensorflow
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
    GIT_TAG d5b57ca93e506df258271ea00fc29cf98383a374 # v2.11.0
)

FetchContent_MakeAvailable(tensorflow)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL cortexa72)
    set(CMAKE_SYSTEM_PROCESSOR aarch64) # TO SOLVE: https://github.com/microsoft/onnxruntime/issues/8425
endif()

# Print value of cmake_system_processor
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TFLITE_ENABLE_XNNPACK OFF)

# set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed -ldl -Wall -Wextra")
set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed -ldl")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(
    "${tensorflow_SOURCE_DIR}/tensorflow/lite"
    "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
    EXCLUDE_FROM_ALL
)

# Uncomment for debug purposes
# message(WARNING "tensorflow_SOURCE_DIR: ${tensorflow_SOURCE_DIR}")
# message(WARNING "subdir1: ${tensorflow_SOURCE_DIR}/tensorflow/lite")
# message(ERROR "subdir2: ${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite")
add_library(${WRAP_LIB_NAME}
    src/tflitewrapper.cpp
)
target_link_libraries(${WRAP_LIB_NAME}
    tensorflow-lite
)

# tflite_with_xnnpack=true
target_compile_definitions(${WRAP_LIB_NAME} PRIVATE tflite_with_xnnpack=false)

# add_custom_target(combined ALL
# COMMAND ${CMAKE_AR} rc libcombined.a $<TARGET_FILE:${WRAP_LIB_NAME}> $<TARGET_FILE:tensorflow-lite>)
add_executable(tflite_test_base
    src/test/test_base.cpp
)
target_link_libraries(tflite_test_base
    ${WRAP_LIB_NAME}
)

add_executable(tflite_test_2d
    src/test/test_2d.cpp
)
target_link_libraries(tflite_test_2d
    ${WRAP_LIB_NAME}
)

add_executable(tflite_test_bulk
    src/test/test_bulk.cpp
)
target_link_libraries(tflite_test_bulk
    ${WRAP_LIB_NAME}
)
