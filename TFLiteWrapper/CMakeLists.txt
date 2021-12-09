cmake_minimum_required(VERSION 3.16)
project(TestTFliteWrapper C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)

if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/tensorflow-src"
    ABSOLUTE
  )
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

# message(WARNING "TENSORFLOW_SOURCE_DIR: ${TENSORFLOW_SOURCE_DIR}")
# message(WARNING "subdir1: ${TENSORFLOW_SOURCE_DIR}/tensorflow/lite")
# message(SEND_ERROR "subdir2: ${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite")

set(CMAKE_CXX_STANDARD 11)
add_library(tflitewrapper
  src/classifier-lib.cpp
)
target_link_libraries(tflitewrapper
  tensorflow-lite
)



add_executable(TestTFliteWrapper
  src/test/main.cpp
)
target_link_libraries(TestTFliteWrapper
    LibraryFile
)