# Description:
#   TensorFlow Lite timbre inference

load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "timbreinference",
    srcs = [
        "classifier-lib.cpp",
        "classifier-lib.hpp"
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)
