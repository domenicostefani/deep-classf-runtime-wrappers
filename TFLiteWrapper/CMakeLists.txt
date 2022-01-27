cmake_minimum_required(VERSION 3.16)
project(TestTFliteWrapper C CXX)

include(FetchContent)

FetchContent_Declare(
  tensorflow
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG        c256c071bb26e1e13b4666d1b3e229e110bc914a # v2.7.0
)
FetchContent_MakeAvailable(tensorflow)


add_subdirectory(
  "${tensorflow_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

# Uncomment for debug purposes
# message(WARNING "tensorflow_SOURCE_DIR: ${tensorflow_SOURCE_DIR}")
# message(WARNING "subdir1: ${tensorflow_SOURCE_DIR}/tensorflow/lite")
# message(WARNING "subdir2: ${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite")

set(CMAKE_CXX_STANDARD 11)
add_library(tflitewrapper
  src/tflitewrapper.cpp
)
target_link_libraries(tflitewrapper
  tensorflow-lite
)



add_executable(TestTFliteWrapper
    src/test/main.cpp
)
target_link_libraries(TestTFliteWrapper
    tflitewrapper
)