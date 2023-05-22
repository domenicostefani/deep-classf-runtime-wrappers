#!/bin/bash

# Cross-compilation script for ElkOS on RPI4
# Instructions at:
# https://elk-audio.github.io/elk-docs/html/documents/building_plugins_for_elk.html#cross-compiling-juce-plugin

mkdir -p build-aarch64
cd build-aarch64

unset LD_LIBRARY_PATH

# Download and install from https://github.com/elk-audio/elkpi-sdk
# source /opt/elk/1.0/environment-setup-aarch64-elk-linux     # Release 0.9.0
source /opt/elk/0.11.0/environment-setup-cortexa72-elk-linux # Release 0.11.0

cmake -DCMAKE_BUILD_TYPE=Release ..
export CXXFLAGS="-O3 -pipe -ffast-math -feliminate-unused-debug-types -funroll-loops"
AR=aarch64-elk-linux-ar make -j$(nproc) CONFIG=Release CFLAGS="-DJUCE_HEADLESS_PLUGIN_CLIENT=1 -Wno-psabi" TARGET_ARCH="-mcpu=cortex-a72 -mtune=cortex-a72"
