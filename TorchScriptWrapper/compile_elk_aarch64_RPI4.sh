#!/usr/bin/env bash

set -e # exit when any command fails

mkdir -p build-aarch64 # Create build directory
cd build-aarch64       # Enter build directory

BUILD_TYPE='Release'

unset LD_LIBRARY_PATH
source /opt/elk/1.0/environment-setup-aarch64-elk-linux
export CXXFLAGS="-O3 -pipe -ffast-math -feliminate-unused-debug-types -funroll-loops"
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE
AR=aarch64-elk-linux-ar make -j$(nproc) CONFIG=$BUILD_TYPE CFLAGS="-DJUCE_HEADLESS_PLUGIN_CLIENT=1 -Wno-psabi" TARGET_ARCH="-mcpu=cortex-a72 -mtune=cortex-a72"
