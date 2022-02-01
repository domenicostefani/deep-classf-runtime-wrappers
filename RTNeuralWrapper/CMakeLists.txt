cmake_minimum_required(VERSION 3.16)

project(rtneuralwrapper VERSION 0.1.0)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Defaulting to RELEASE build type")
endif()

set(CMAKE_CXX_FLAGS " -DEIGEN_STACK_ALLOCATION_LIMIT=0")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(USE_COMPILE_TIME_API)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DUSE_COMPILE_TIME_API")
    set(NNLOADTYPE "ctime")
    message(STATUS "Compiling wrapper for dynamic run-time model loading")
else()
    set(NNLOADTYPE "rtime")
    message(STATUS "Compiling wrapper for dynamic run-time model loading")
endif()


set(LIB_NAME rtneuralwrapper${NNLOADTYPE})

add_subdirectory(libs/RTNeural)

include(cmake/SIMDExtensions.cmake)
include(cmake/ChooseBackend.cmake)

include_directories(libs/RTNeural)

ADD_LIBRARY(${LIB_NAME} STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src/rtneuralwrapper.cpp)

target_link_libraries(${LIB_NAME} LINK_PUBLIC RTNeural)

# CMake instructions to test using the lib (Dynamic run-time NN loading)
SET( APP_EXE rtneural_${NNLOADTYPE}_test_base )

ADD_EXECUTABLE( ${APP_EXE}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/test/test1.cpp ) 

TARGET_LINK_LIBRARIES( ${APP_EXE}
                       ${LIB_NAME} )

