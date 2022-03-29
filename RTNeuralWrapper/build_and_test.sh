#!/usr/bin/env bash

set -e      # exit when any command fails


mkdir -p  build # Create build directory
cd build    # Enter build directory

cmake .. -DUSE_COMPILE_TIME_API=false   # Run cmake, compiling the library that dynamically loads the neural network model
make        # Build library and test

# Run basic test
./rtneural_rtime_test_base ../data/model_weights.json
echo "test run-time model [1/2] Ok"

# # Run bulk test
# ./tflite_test_bulk ../data/model_test1.tflite ../data/bulk/small_features.csv ../data/bulk/small_labels.csv
# echo "test run-time model [2/2] Ok"



cmake .. -DUSE_COMPILE_TIME_API=true   # Run cmake, compiling the library that statically defines the neural network model
make        # Build library and test

# Run basic test
./rtneural_ctime_test_base ../data/model_weights.json
echo "test compile-time model [1/2] Ok"

# # Run bulk test
# ./tflite_test_bulk ../data/model_test1.tflite ../data/bulk/small_features.csv ../data/bulk/small_labels.csv
# echo "test compile-time model [2/2] Ok"



cd ..
mkdir -p bin
cd bin
rm -rf *
ln -s ../build/librtneuralwrapperctime.a .
ln -s ../build/librtneuralwrapperrtime.a .
ln -s ../src/rtneuralwrapper.h .
mkdir -p test
cd test
ln -s ../../build/rtneural_ctime_test_base .
# ln -s ../../build/rtneural_ctime_test_bulk .
ln -s ../../build/rtneural_rtime_test_base .
# ln -s ../../build/rtneural_rtime_test_bulk .

echo "Links to the binaries are in ./bin/"