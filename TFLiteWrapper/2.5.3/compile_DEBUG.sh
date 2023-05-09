#!/usr/bin/env bash

set -e      # exit when any command fails

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=$SCRIPT_DIR/data
BUILD_DIR=$SCRIPT_DIR/build
SRC_DIR=$SRC_DIR
BIN_DIR=$SCRIPT_DIR/bin

mkdir -p  $BUILD_DIR # Create build directory
cd $BUILD_DIR    # Enter build directory

cmake ..  -DCMAKE_BUILD_TYPE=Debug 

make -j`nproc` CONFIG=Debug       # Build library and test