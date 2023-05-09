#!/bin/bash

unset LD_LIBRARY_PATH

# Source the environment setup script for the correct ElkOS version

# source /opt/elk/1.0/environment-setup-aarch64-elk-linux # elk 0.9.0
source /opt/elk/0.11.0/environment-setup-cortexa72-elk-linux # Elk 0.11.0

mkdir -p build-aarch64
cd build-aarch64
cmake .. -DTFLITE_ENABLE_XNNPACK=OFF
CMAKE_SYSTEM_PROCESSOR aarch64

export CXXFLAGS=CXXFLAGS:"-O3 -pipe -ffast-math -feliminate-unused-debug-types -funroll-loops -Wfatal-errors"
AR=aarch64-elk-linux-ar make -j$(nproc) CONFIG=Release CFLAGS="-DJUCE_HEADLESS_PLUGIN_CLIENT=1 -Wno-psabi -Wfatal-errors" TARGET_ARCH="-mcpu=cortex-a72 -mtune=cortex-a72"file
