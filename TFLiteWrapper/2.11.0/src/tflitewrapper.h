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

/**
 * @brief Get the Model Input Size for 1dimentional input models
 * 
 * @param cls 
 * @return size_t 
 */
size_t getModelInputSize1d(ClassifierPtr cls);

/**
 * @brief Get the Model Input Size2d object
 * 
 * @param cls 
 * @param rows 
 * @param columns 
 */
void getModelInputSize2d(ClassifierPtr cls, size_t& rows, size_t& columns);

/**
 * @brief Get the Model Output size
 * 
 * @param cls 
 * @return size_t 
 */
size_t getModelOutputSize(ClassifierPtr cls);

/**
 * @brief Dynamically allocate an instance of a classifier object (do not use in real time threads!) 
 * 
 * @param filename path to the tflite model file
 * @param verbose  verbose mode (to disable in real time threads)
 * @return ClassifierPtr 
 */
ClassifierPtr createClassifier(const std::string &filename, bool verbose = false);

/**
 * @brief Free the classifier memory (do not use in real time threads)
 * 
 * @param cls pointer to the classifier object
 */
void deleteClassifier(ClassifierPtr cls);

/**
 * @brief  Feed a feature array (C Array) to the model, perform inference and return the prediction 
 * This function is used to classify a feature vector. The feature vector is passed as a pointer to a float (c array).
 * 
 * @param cls 
 * @param featureVector 
 * @param numFeatures 
 * @param outputVector 
 * @param numClasses 
 * @param verbose 
 * @return int 
 */
int classify(ClassifierPtr cls, const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses, bool verbose = false);

/**
 * @brief Feed a feature array (C++ std Array) to the model, perform inference and return the prediction 
 * This function is used to classify a feature vector. The feature vector is passed as a std::array<float,IN_SIZE>.
 * The function does not allocate memory, it just passes the pointer to the first element of the array with the .data() method.
 * 
 * @tparam IN_SIZE 
 * @tparam OUT_SIZE 
 * @param cls 
 * @param featureArray 
 * @param outputArray 
 * @return int  Classification result
 */
template<std::size_t IN_SIZE, std::size_t OUT_SIZE>
int classify(ClassifierPtr cls, std::array<float,IN_SIZE>& featureArray, std::array<float,OUT_SIZE>& outputArray) {
    return classify(cls, featureArray.data() , (size_t)IN_SIZE, outputArray.data(), (size_t)OUT_SIZE);
}

/**
 * @brief Feed a feature array (C++ std Vector) to the model, perform inference and return the prediction
 * This function is used to classify a feature vector. The feature vector is passed as a std::vector<float>.
 * This is particularly useful when the input size is not known at compile time, expecially for test code.
 * A vector of random test data can be created with the help of getModelInputSize1d and passed to this function.
 * 
 * @param cls           Classifier object
 * @param featureVector Input vector
 * @param outputVector  Output vector
 * @return int          Classification result
 */
int classify(ClassifierPtr cls, std::vector<float>& featureVector, std::vector<float>& outputVector);


/**
 * @brief Classify a 2D matrix stored in a flat array
 * 
 * @param cls 
 * @param flatFeatureMatrix 
 * @param nRows 
 * @param nCols 
 * @param outputVector 
 * @param numClasses 
 * @param verbose 
 * @return int 
 */
int classifyFlat2D(ClassifierPtr cls, const float flatFeatureMatrix[], size_t nRows, size_t nCols, float outputVector[], size_t numClasses, bool verbose = false);

/**
 * @brief Classify a 2D matrix stored in a flat std::array
 * 
 * @tparam N_ROWS         Number of rows in the input matrix
 * @tparam N_COLS         Number of columns in the input matrix
 * @tparam OUT_SIZE       Number of classes in the output vector
 * @param cls             Classifier object
 * @param flatInputMatrix Input matrix
 * @param outputArray     Output vector
 * @return int            Classification result
 */
template<std::size_t N_ROWS, std::size_t N_COLS, std::size_t OUT_SIZE>
int classifyFlat2D(ClassifierPtr cls, std::array<float,N_ROWS * N_COLS>& flatInputMatrix, std::array<float,OUT_SIZE>& outputArray, bool verbose = false) {
    return classifyFlat2D(cls, flatInputMatrix.data(), N_ROWS, N_COLS, outputArray.data(), (size_t)OUT_SIZE, verbose);
}

/**
 * @brief Classify a 2D matrix stored in a flat vector
 * 
 * @param cls             Classifier object
 * @param flatInputMatrix Input matrix
 * @param nRows           Number of rows in the input matrix
 * @param nCols           Number of columns in the input matrix
 * @param outputVector    Output vector
 * @return int            Classification result
 */
int classifyFlat2D(ClassifierPtr cls, std::vector<float>& flatInputMatrix, size_t nRows, size_t nCols, std::vector<float>& outputVector, bool verbose = false);