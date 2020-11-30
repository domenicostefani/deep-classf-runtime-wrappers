#!/bin/bash

aarch64-linux-gnu-g++ -o inference_linux_aarch64 -L/home/domenico/Develop/timbreInference/compiled_library -I/home/domenico/Develop/timbreInference/compiled_library -L/home/domenico/Develop/timbreInference/tensorflow-src/tensorflow/lite/tools/make/gen/linux_aarch64/lib main.cpp -pthread -lliteclassifier -ltensorflow-lite

