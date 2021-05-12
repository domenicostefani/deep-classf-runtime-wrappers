#!/bin/bash

TENSORFLOWBASE="/home/cimil-01/Develop/tensorflow/"

header1="-I"$TENSORFLOWBASE
header2="-I"$TENSORFLOWBASE"tensorflow/lite/tools/make/downloads/absl"
header3="-I"$TENSORFLOWBASE"tensorflow/lite/tools/make/downloads/flatbuffers/include"

aarch64-linux-gnu-g++ -c -fPIC -o./your_library_here/classifier-lib.o $header1 $header2 $header3 your_library_here/classifier-lib.cpp

#Create the .a library file
mkdir -p compiled_library
ar rvs compiled_library/libliteclassifier.a your_library_here/classifier-lib.o

cp ./your_library_here/classifier-lib.hpp ./compiled_library/liteclassifier.h
