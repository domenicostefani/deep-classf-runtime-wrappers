#!/bin/bash

TENSORFLOWBASE="/home/cimil-01/Develop/tensorflow/"

header1="-I"$TENSORFLOWBASE
header2="-I"$TENSORFLOWBASE"tensorflow/lite/tools/make/downloads/absl"
header3="-I"$TENSORFLOWBASE"tensorflow/lite/tools/make/downloads/flatbuffers/include"

g++ -c -fPIC -o./your_library_here/classifier-lib.o $header1 $header2 $header3 your_library_here/classifier-lib.cpp

#Create the .a library file
mkdir -p the_compiled_library
ar rvs the_compiled_library/libliteclassifier.a your_library_here/classifier-lib.o

cp ./your_library_here/classifier-lib.hpp ./the_compiled_library/liteclassifier.h
