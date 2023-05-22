/*
==============================================================================*/
#include "rtneuralwrapper.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>  // std::numeric_limits
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include "RTNeural.h"

#ifdef USE_COMPILE_TIME_API
// --- Model B ---
typedef RTNeural::ModelT<float, 173, 8,
                         RTNeural::DenseT<float, 173, 350>,
                         RTNeural::ReLuActivationT<float, 350>,
                         RTNeural::DenseT<float, 350, 350>,
                         RTNeural::ReLuActivationT<float, 350>,
                         RTNeural::DenseT<float, 350, 350>,
                         RTNeural::ReLuActivationT<float, 350>,
                         RTNeural::DenseT<float, 350, 350>,
                         RTNeural::ReLuActivationT<float, 350>,
                         RTNeural::DenseT<float, 350, 350>,
                         RTNeural::ReLuActivationT<float, 350>,
                         RTNeural::DenseT<float, 350, 350>,
                         RTNeural::ReLuActivationT<float, 350>,
                         RTNeural::DenseT<float, 350, 8>>
    model_t;

// --- Model C ---
// typedef RTNeural::ModelT<float, 173, 8,
//         RTNeural::DenseT<float, 173, 350>,
//         RTNeural::ReLuActivationT<float, 350>,
//         RTNeural::DenseT<float, 350, 8>> model_t;

typedef model_t *model_ptr;
#else
typedef RTNeural::Model<float> model_t;
typedef std::unique_ptr<model_t, std::default_delete<model_t>> model_ptr;
#endif

// Definition of the classifier class
class Classifier {
public:
    /** Constructor */
    Classifier(const std::string &filename, bool verbose = false);
    /** Destructor */
    ~Classifier();
    /** Internal classification function, called by wrappers */
    int classify_internal(const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses);

private:
    /** Load the .onnx model and create inference session */
    model_ptr loadModel(const std::string &filename, bool verbose = false);

    /** ind the index of the maximum value in an array */
    int argmax(const float vec[], size_t vecSize) const;

    //--------------------------------------------------------------------------
    model_ptr model;

    size_t inputTensorSize = 173;
    size_t outputTensorSize = 8;
    std::vector<float> inputTensorValues;
};

Classifier::Classifier(const std::string &filename, bool verbose) {
    // Load model
    if (verbose) {
        std::cout << std::setfill('-') << std::setw(40) << "" << std::endl;
#ifdef USE_COMPILE_TIME_API
        std::cout << "Parsing model (compile-time defined model, Model B)..." << std::endl;
#else
        std::cout << "Parsing model (dynamic model loading)..." << std::endl;
#endif
    }
    this->model = loadModel(filename, verbose);
    if (verbose) {
        std::cout << "Model loaded successfully." << std::endl;
        std::cout << "File: " << filename << std::endl;
    }

    /* implementation for dynamic model loading
    inputTensorSize = this->model->layers.front()->in_size;
    outputTensorSize = this->model->layers.back()->out_size;
    */

    this->model->reset();

    inputTensorValues = std::vector<float>(inputTensorSize);

    // Prime the classifier
    std::vector<float> pIv(inputTensorSize);
    std::vector<float> pOv(outputTensorSize);

    this->classify_internal(&pIv[0], pIv.size(), &pOv[0], pOv.size());
    /*
     * The priming operation should ensure that every allocation performed
     * by the Run method is perfomed here and not in the real-time thread.
     */
}

Classifier::~Classifier() {
#ifdef USE_COMPILE_TIME_API
    delete this->model;
#endif
}

int Classifier::classify_internal(const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses) {
    if (numFeatures != inputTensorSize)
        throw std::logic_error("Error, input vector has to have size: " + std::to_string(inputTensorSize) + " (Found " + std::to_string(numFeatures) + " instead)");

    // Fill `input`.
    for (size_t i = 0; i < numFeatures; ++i)
        inputTensorValues[i] = featureVector[i];

    // Run inference
    this->model->forward(inputTensorValues.data());

    if (numClasses != outputTensorSize)
        throw std::logic_error("Error, output vector has to have size: " + std::to_string(outputTensorSize) + " (Found " + std::to_string(numClasses) + " instead)");

    // Copy output and save max
    for (size_t i = 0; i < numClasses; ++i)
        outputVector[i] = this->model->getOutputs()[i];

    return argmax(outputVector, numClasses);
}

model_ptr Classifier::loadModel(const std::string &filename, bool verbose) {
    std::ifstream jsonStream(filename, std::ifstream::binary);
#ifdef USE_COMPILE_TIME_API
    auto modelT = new model_t;
    modelT->parseJson(jsonStream, verbose);
    return modelT;
#else
    auto model = RTNeural::json_parser::parseJson<float>(jsonStream, true);
    return model;
#endif
}

int Classifier::argmax(const float vec[], size_t vecSize) const {
    float max = std::numeric_limits<float>::lowest();
    int argmax = -1;
    for (size_t i = 0; i < vecSize; ++i) {
        if (vec[i] > max) {
            argmax = i;
            max = vec[i];
        }
    }
    return argmax;
}

/***** Handle functions *****/
ClassifierPtr createClassifier(const std::string &filename, bool verbose) {
#ifdef USE_COMPILE_TIME_API
    std::cout << "I am compile time optimized" << std::endl
              << std::flush;
#else
    std::cout << "I am run time optimized" << std::endl
              << std::flush;
#endif

    return new Classifier(filename, verbose);
}

void deleteClassifier(ClassifierPtr cls) {
    if (cls)
        delete cls;
}

int classify(ClassifierPtr cls, const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses) {
    return cls->classify_internal(featureVector, numFeatures, outputVector, numClasses);
}

void softmax(float logitsArray[], size_t numClasses, bool verbose) {
    if (verbose)
        std::cout << "Applying softmax..." << std::endl
                  << std::flush;

    // Subtract Max from logits for stable Softmax https://stackoverflow.com/a/49212689 (TF does this too)
    float max = logitsArray[0];
    for (size_t i = 1; i < numClasses; ++i)
        if (logitsArray[i] > max)
            max = logitsArray[i];
    for (size_t i = 0; i < numClasses; ++i)
        logitsArray[i] -= max;

    float tsum = 0;
    for (size_t i = 0; i < numClasses; ++i)
        tsum += exp(logitsArray[i]);
    for (size_t i = 0; i < numClasses; ++i)
        logitsArray[i] = exp(logitsArray[i]) / tsum;

    if (verbose)
        std::cout << "Done." << std::endl
                  << std::flush;
}