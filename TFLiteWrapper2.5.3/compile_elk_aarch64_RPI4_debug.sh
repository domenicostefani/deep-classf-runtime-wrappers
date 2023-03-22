#!/bin/bash

unset LD_LIBRARY_PATH
source /opt/elk/1.0/environment-setup-aarch64-elk-linux

mkdir -p build-elk
cd build-elk
cmake .. -DTFLITE_ENABLE_XNNPACK=OFF

export CXXFLAGS=CXXFLAGS:"-O3 -pipe -ffast-math -feliminate-unused-debug-types -funroll-loops -Wfatal-errors"
AR=aarch64-elk-linux-ar make -j`nproc` CONFIG=Debug CFLAGS="-DJUCE_HEADLESS_PLUGIN_CLIENT=1 -Wno-psabi -Wfatal-errors" TARGET_ARCH="-mcpu=cortex-a72 -mtune=cortex-a72"file 