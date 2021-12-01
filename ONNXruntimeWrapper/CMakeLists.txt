cmake_minimum_required(VERSION 3.0.0)

set(LIB_NAME onnxclassifier)

project(${LIB_NAME} VERSION 0.1.0)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Defaulting to RELEASE build type")
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(libs/onnxruntime/include)

ADD_LIBRARY( ${LIB_NAME} STATIC
             ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx-classifier-lib.cpp )

target_link_libraries(${LIB_NAME} libs/onnxruntime/lib/libonnxruntime.so)


# CMake instructions to test using the static lib

SET( APP_EXE test-onnx-lib )

ADD_EXECUTABLE( ${APP_EXE}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/testlib.cpp ) 

TARGET_LINK_LIBRARIES( ${APP_EXE}
                        ${LIB_NAME} )
