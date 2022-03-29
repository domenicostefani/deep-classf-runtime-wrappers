cmake_minimum_required(VERSION 3.0.0)

set(LIB_NAME torchscriptwrapper)

project(${LIB_NAME} VERSION 0.1.0)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Defaulting to RELEASE build type")
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")



# # change path to libtorch location
# set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch/")

# message(STATUS "Torch dir selected: ${CMAKE_PREFIX_PATH}")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(SYSTEM libs/libtorch/include)


ADD_LIBRARY( ${LIB_NAME} STATIC
             ${CMAKE_CURRENT_SOURCE_DIR}/src/torchscriptwrapper.cpp )


target_link_libraries(${LIB_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch/lib/libc10.so 
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch/lib/libtorch_cpu.so 
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch/lib/libtorch.so)
# set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 14)


# CMake instructions to test using the static lib

SET( APP_EXE test-torch-lib )

ADD_EXECUTABLE( ${APP_EXE}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/test/test1.cpp ) 

TARGET_LINK_LIBRARIES( ${APP_EXE}
                        ${LIB_NAME} )




SET( APP_EXE testmeasure-torch )

ADD_EXECUTABLE( ${APP_EXE}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/test/testmeasure.cpp ) 

TARGET_LINK_LIBRARIES( ${APP_EXE}
                        ${LIB_NAME} )
