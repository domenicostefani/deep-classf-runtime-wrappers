cmake_minimum_required(VERSION 3.0.0)

set(LIB_NAME torchclassifier)

project(${LIB_NAME} VERSION 0.1.0)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Defaulting to RELEASE build type")
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")



# change path to libtorch location
set(CMAKE_PREFIX_PATH "/home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper/libs/libtorch/")

message(STATUS "Torch dir selected: ${CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

ADD_LIBRARY( ${LIB_NAME} STATIC
             ${CMAKE_CURRENT_SOURCE_DIR}/src/torch-classifier-lib.cpp )

target_link_libraries(${LIB_NAME} "${TORCH_LIBRARIES}")
# set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 14)


# CMake instructions to test using the static lib

SET( APP_EXE test-torch-lib )

ADD_EXECUTABLE( ${APP_EXE}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/testlib.cpp ) 

TARGET_LINK_LIBRARIES( ${APP_EXE}
                        ${LIB_NAME} )
