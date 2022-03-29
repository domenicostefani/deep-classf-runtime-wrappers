#!/usr/bin/env bash

set -e      # exit when any command fails


mkdir -p  build # Create build directory
cd build    # Enter build directory




unset LD_LIBRARY_PATH
source /opt/elk/1.0/environment-setup-aarch64-elk-linux
export CXXFLAGS="-O3 -pipe -ffast-math -feliminate-unused-debug-types -funroll-loops"
cmake .. -DUSE_COMPILE_TIME_API=false   # Run cmake, compiling the library that dynamically loads the neural network model
AR=aarch64-elk-linux-ar make -j`nproc` CONFIG=Release CFLAGS="-DJUCE_HEADLESS_PLUGIN_CLIENT=1 -Wno-psabi" TARGET_ARCH="-mcpu=cortex-a72 -mtune=cortex-a72"



unset LD_LIBRARY_PATH
source /opt/elk/1.0/environment-setup-aarch64-elk-linux
export CXXFLAGS="-O3 -pipe -ffast-math -feliminate-unused-debug-types -funroll-loops"
cmake .. -DUSE_COMPILE_TIME_API=true   # Run cmake, compiling the library that statically defines the neural network model
AR=aarch64-elk-linux-ar make -j`nproc` CONFIG=Release CFLAGS="-DJUCE_HEADLESS_PLUGIN_CLIENT=1 -Wno-psabi" TARGET_ARCH="-mcpu=cortex-a72 -mtune=cortex-a72"



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