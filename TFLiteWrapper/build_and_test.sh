#!/usr/bin/env bash

set -e      # exit when any command fails


mkdir -p  build # Create build directory
cd build    # Enter build directory

cmake ..    # Run Cmake, which pulls tensorflow v2.7.0 from github

make        # Build library and test

# Run basic test
./tflite_test_base ../data/model_test1.tflite
echo "test [1/2] Ok"

# Run bulk test
./tflite_test_bulk ../data/model_test1.tflite ../data/bulk/small_features.csv ../data/bulk/small_labels.csv
echo "test [2/2] Ok"

cd ..
mkdir -p bin
cd bin
rm -rf libtflitewrapper.a tflitewrapper.h
ln -s ../build/libtflitewrapper.a .
ln -s ../src/tflitewrapper.h .
mkdir -p test
cd test
ln -s ../../build/tflite_test_base .
ln -s ../../build/tflite_test_bulk .

echo "Links to the binaries are in ./bin/"