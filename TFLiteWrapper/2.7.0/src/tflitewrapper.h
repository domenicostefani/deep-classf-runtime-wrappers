/*
 * TFlite Classifier library
 * Author: Domenico Stefani (domenico.stefani96@gmail.com)
 *
 * This header exposes the functions used to run inference with TFLite.
 * To see how to use it, check test_library/main.cpp
 *
 * The structure is a bit convoluted since it used an opaque pointer (to avoid having to include many headers later)
 *
 */
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <limits>  // std::numeric_limits
#include <string>
#include <utility>
#include <vector>

class Classifier;                   // Forward definition of the Classifier class
using ClassifierPtr = Classifier*;  // Opaque pointer for classifier object

/** Dynamically allocate an instance of a classifier object (do not use in real time threads!) */
ClassifierPtr createClassifier(const std::string& filename, bool verbose = false);

/** Feed a feature array (C Array) to the model, perform inference and return the prediction */
int classify(ClassifierPtr cls, const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses, bool verbose = false);

/** Feed a feature array (C++ std Array) to the model, perform inference and return the prediction */
template <std::size_t IN_SIZE, std::size_t OUT_SIZE>
int classify(ClassifierPtr cls, std::array<float, IN_SIZE>& featureArray, std::array<float, OUT_SIZE>& outputArray) {
    const float* fa = &(featureArray[0]);
    float* oa = &(outputArray[0]);
    return classify(cls, fa, (size_t)IN_SIZE, oa, (size_t)OUT_SIZE);
}

/** Free the classifier memory (do not use in real time threads) */
void deleteClassifier(ClassifierPtr cls);

/**
 * @brief Apply softmax to a logits array
 * Apply softmax to a logits array when using networks that do not have a softmax output layer
 *
 * @param logitsArray Array of logits
 * @param numClasses Number of classes (or size of the array)
 * @param verbose Whether to print debug messages
 */
void softmax(float logitsArray[], size_t numClasses, bool verbose = true);