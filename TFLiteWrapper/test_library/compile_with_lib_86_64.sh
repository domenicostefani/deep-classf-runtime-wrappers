#!/bin/bash

g++ -o inference_x86_64 -L/home/domenico/Develop/timbreInference/compiled_library -I/home/domenico/Develop/timbreInference/compiled_library -L/home/domenico/Develop/timbreInference/tensorflow-src/tensorflow/lite/tools/make/gen/linux_x86_64/lib main.cpp -pthread -lliteclassifier -ltensorflow-lite

