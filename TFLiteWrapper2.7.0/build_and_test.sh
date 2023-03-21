#!/usr/bin/env bash

set -e      # exit when any command fails

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=$SCRIPT_DIR/data
BUILD_DIR=$SCRIPT_DIR/build
SRC_DIR=$SRC_DIR
BIN_DIR=$SCRIPT_DIR/bin

mkdir -p  $BUILD_DIR # Create build directory
cd $BUILD_DIR    # Enter build directory

cmake ..    # Run Cmake, which pulls tensorflow v2.7.0 from github

make        # Build library and test

# Run basic test
./tflite_test_base $DATA_DIR/test_model.tflite
echo "test [1/2] Ok"

# Run bulk test
./tflite_test_bulk $DATA_DIR/test_model.tflite $DATA_DIR/bulk/small_features.csv $DATA_DIR/bulk/small_labels.csv
echo "test [2/2] Ok"

# Run basic test but with model with 2D convolutions, flat input and reshaping layers
echo $(realpath $DATA_DIR/test_model_2d.tflite)
./tflite_test_base $DATA_DIR/test_model_2d.tflite


# Create links to the binaries
cd ..
mkdir -p $BIN_DIR
cd $BIN_DIR
rm -rf libtflitewrapper.a tflitewrapper.h
ln -s $BUILD_DIR/libtflitewrapper.a .
ln -s $SRC_DIR/tflitewrapper.h .
mkdir -p test
cd test
rm tflite_test_base
rm tflite_test_bulk
ln -s $BUILD_DIR/tflite_test_base .
ln -s $BUILD_DIR/tflite_test_bulk .

echo "Links to the binaries are in $BIN_DIR"