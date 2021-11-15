cmake_minimum_required(VERSION 3.0.0)

set(LIB_NAME libtorchclassifier)

project(${LIB_NAME} VERSION 0.1.0)

# change path to libtorch location
set(CMAKE_PREFIX_PATH "D:/cpplibs/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

ADD_LIBRARY( ${LIB_NAME} STATIC
             your_library_here/classifier-lib.cpp )

target_link_libraries(${LIB_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 14)


# CMake instructions to test using the static lib

SET( APP_EXE StaticTest )

ADD_EXECUTABLE( ${APP_EXE}
                test_library/main.cpp ) 

TARGET_LINK_LIBRARIES( ${APP_EXE}
                        ${LIB_NAME} )