#!/bin/bash

# Cross-compilation script for ElkOS on RPI4
# Instructions at:
# https://elk-audio.github.io/elk-docs/html/documents/building_plugins_for_elk.html#cross-compiling-juce-plugin

mkdir -p build-x86-64
cd build-x86-64

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc) CONFIG=Release
