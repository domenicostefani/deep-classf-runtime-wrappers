#!/bin/bash
BASE="/home/cimil-01/Develop/timbreInference"
TFLITE_LIB="/home/cimil-01/Develop/tflite-build-master20210510/linux_aarch64/lib"
aarch64-linux-gnu-g++ -g -o inference_linux_aarch64 -L$BASE/the_compiled_library -I$BASE/the_compiled_library -L$TFLITE_LIB main.cpp -pthread -lliteclassifier -ltensorflow-lite -Wl,--no-as-needed -ldl

