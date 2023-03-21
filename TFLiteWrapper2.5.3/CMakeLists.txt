cmake_minimum_required(VERSION 3.16)
project(TFliteWrapper C CXX)
set(WRAP_LIB_NAME tflitewrapper)

include(FetchContent)
# FetchContent_Declare(
#   tensorflow
#   GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
#   GIT_TAG        85c8b2a817f95a3e979ecd1ed95bff1dc1335cff # v2.4.1
# )

# FetchContent_Declare(
#   tensorflow
#   GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
#   GIT_TAG        64918868e2154b06c7479347a59a4230f785e9fa # v2.4.4
# )

FetchContent_Declare(
  tensorflow
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG        959e9b2a0c06df945f9fb66bd367af8832ca0d28 # v2.5.3
)
  

FetchContent_MakeAvailable(tensorflow)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed -ldl -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(
  "${tensorflow_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

# Uncomment for debug purposes
# message(WARNING "tensorflow_SOURCE_DIR: ${tensorflow_SOURCE_DIR}")
# message(WARNING "subdir1: ${tensorflow_SOURCE_DIR}/tensorflow/lite")
# message(ERROR "subdir2: ${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite")

set(CMAKE_CXX_STANDARD 11)
add_library(${WRAP_LIB_NAME}
  src/tflitewrapper.cpp
)
target_link_libraries(${WRAP_LIB_NAME}
  tensorflow-lite
)
# add_custom_target(combined ALL
#    COMMAND ${CMAKE_AR} rc libcombined.a $<TARGET_FILE:${WRAP_LIB_NAME}> $<TARGET_FILE:tensorflow-lite>)


add_executable(tflite_test_base
    src/test/test_base.cpp
)
target_link_libraries(tflite_test_base
    ${WRAP_LIB_NAME}
)

add_executable(tflite_test_bulk
    src/test/test_bulk.cpp
)
target_link_libraries(tflite_test_bulk
    ${WRAP_LIB_NAME}
)
