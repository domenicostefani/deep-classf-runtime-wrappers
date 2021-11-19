/*
 * torchscript wrapper library
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
int classify(ClassifierPtr cls, const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses);

/** Feed a feature array (C++ std Array) to the model, perform inference and return the prediction */
template<std::size_t IN_SIZE, std::size_t OUT_SIZE>
int classify(ClassifierPtr cls, std::array<float,IN_SIZE>& featureArray, std::array<float,OUT_SIZE>& outputArray)
{
    const float* fa = &(featureArray[0]);
    float* oa = &(outputArray[0]);
    return classify(cls, fa , (size_t)IN_SIZE, oa, (size_t)OUT_SIZE);
}

/** Free the classifier memory (do not use in real time threads) */
void deleteClassifier(ClassifierPtr cls);





