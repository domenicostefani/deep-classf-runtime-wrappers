cmake_minimum_required(VERSION 3.10)
project(TFliteWrapper)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


ADD_LIBRARY( liteclassifier STATIC
             src/classifier-lib.cpp )

set(TENSORFLOW_SOURCE_DIR "/home/cimil-01/Develop/tensorflow" CACHE PATH
  "Directory that contains the TensorFlow project"
)
             

target_include_directories(liteclassifier PRIVATE ${TENSORFLOW_SOURCE_DIR})

# CMake instructions to test using the static lib

SET( APP_EXE StaticTest )

ADD_EXECUTABLE( ${APP_EXE}
                test_library/main.cpp) 

TARGET_LINK_LIBRARIES( ${APP_EXE}
                       MyStaticLib )
