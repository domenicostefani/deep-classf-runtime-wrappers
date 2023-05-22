#!/bin/bash

unset LD_LIBRARY_PATH

BUILD_FOLDER="build-amd64"
mkdir -p $BUILD_FOLDER
cd $BUILD_FOLDER
cmake .. -DTFLITE_ENABLE_XNNPACK=OFF

make -j$(nproc) CONFIG=Release
