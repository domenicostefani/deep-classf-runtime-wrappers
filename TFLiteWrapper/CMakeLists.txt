cmake_minimum_required(VERSION 3.16)
project(TFliteWrapper C CXX)
set(WRAP_LIB_NAME tflitewrapper)

include(FetchContent)
FetchContent_Declare(
  tensorflow
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG        c256c071bb26e1e13b4666d1b3e229e110bc914a # v2.7.0
)
FetchContent_MakeAvailable(tensorflow)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
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
# message(WARNING "subdir2: ${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite")

set(CMAKE_CXX_STANDARD 11)
add_library(${WRAP_LIB_NAME}
  src/tflitewrapper.cpp
)
target_link_libraries(${WRAP_LIB_NAME}
  tensorflow-lite
)



add_executable(tflite_test_base
    src/test/test1.cpp
)
target_link_libraries(tflite_test_base
    tflitewrapper
)

add_executable(tflite_test_bulk
    src/test/test2.cpp
)
target_link_libraries(tflite_test_bulk
    tflitewrapper
)