/*
 * TFlite Classifier library
 * Author: Domenico Stefani (domenico.stefani96@gmail.com)
*/
#pragma once

#include <string>
#include <vector>

/** Opaque pointer for classifier object */
class Classifier;
typedef Classifier* ClassifierPtr;

/** Dynamically allocate an instance of a classifier object (do not use in real time threads) */
ClassifierPtr createClassifier(const std::string &filename, bool verbose = false);

/** Feed the feature vector to the model, execute and return the prediction */
std::pair<int,float> classify(ClassifierPtr cls, std::vector<float> featureVector, std::string& info, bool verbose = false);

/** Free the classifier memory (do not use in real time threads) */
void deleteClassifier(ClassifierPtr cls);
