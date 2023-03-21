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

#include <string>
#include <vector>
#include <array>
#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <utility>
#include <limits>       // std::numeric_limits

class Classifier;                   // Forward definition of the Classifier class
using ClassifierPtr = Classifier* ; // Opaque pointer for classifier object

/** Dynamically allocate an instance of a classifier object (do not use in real time threads!) */
ClassifierPtr createClassifier(const std::string &filename, bool verbose = false);

/** Feed a feature array (C Array) to the model, perform inference and return the prediction */
int classify(ClassifierPtr cls, const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses, bool verbose = false);

size_t getModelInputSize(ClassifierPtr cls);
size_t getModelOutputSize(ClassifierPtr cls);


/** Feed a feature array (C++ std Array) to the model, perform inference and return the prediction */
template<std::size_t IN_SIZE, std::size_t OUT_SIZE>
int classify(ClassifierPtr cls, std::array<float,IN_SIZE>& featureArray, std::array<float,OUT_SIZE>& outputArray)
{
    return classify(cls, featureArray.data() , (size_t)IN_SIZE, outputArray.data(), (size_t)OUT_SIZE);
}

int classify(ClassifierPtr cls, std::vector<float>& featureVector, std::vector<float>& outputVector);

/** Free the classifier memory (do not use in real time threads) */
void deleteClassifier(ClassifierPtr cls);





