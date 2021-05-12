/*
 * TFlite Classifier library
 * Author: Domenico Stefani (domenico.stefani96@gmail.com)
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
#include <chrono>  //TODO: remove

/** Opaque pointer for classifier object */
class Classifier;
using ClassifierPtr = Classifier* ;

/** Dynamically allocate an instance of a classifier object (do not use in real time threads) */
ClassifierPtr createClassifier(const std::string &filename, bool verbose = false);

// /** Feed the feature vector to the model, execute and return the prediction */
int classify(ClassifierPtr cls, const std::vector<float>& featureVector, std::vector<float>& outVector);

/** Feed the feature array to the model, execute and return the prediction */
// int classify(ClassifierPtr cls, const std::array<float,IN_SIZE>& featureVector, std::array<float,OUT_SIZE>& outVector);

/** Free the classifier memory (do not use in real time threads) */
void deleteClassifier(ClassifierPtr cls);





