cmake_minimum_required(VERSION 3.0.0)

set(LIB_NAME rtneuralclassifier)

project(${LIB_NAME} VERSION 0.1.0)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Defaulting to RELEASE build type")
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "-DEIGEN_STACK_ALLOCATION_LIMIT=0")

add_subdirectory(libs/RTNeural)

include(cmake/SIMDExtensions.cmake)
include(cmake/ChooseBackend.cmake)

include_directories(libs/RTNeural)

ADD_LIBRARY( ${LIB_NAME} STATIC
             ${CMAKE_CURRENT_SOURCE_DIR}/src/RTNeural-classifier-lib.cpp )

target_link_libraries(${LIB_NAME} LINK_PUBLIC RTNeural)

# CMake instructions to test using the static lib

SET( APP_EXE test-RTNeural-lib )

ADD_EXECUTABLE( ${APP_EXE}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/testlib.cpp ) 

TARGET_LINK_LIBRARIES( ${APP_EXE}
                        ${LIB_NAME} )
