#!/usr/bin/env bash

set -e      # exit when any command fails


mkdir -p  build # Create build directory
cd build    # Enter build directory

cmake ..    # Run Cmake, which pulls tensorflow v2.7.0 from github

make        # Build library and test

# Run basic test
./TestTFliteWrapper ../data/model_test1.tflite

cd ..
mkdir -p bin
cd bin
rm -rf libtflitewrapper.a tflitewrapper.h
ln -s ../build/libtflitewrapper.a .
ln -s ../src/tflitewrapper.h .