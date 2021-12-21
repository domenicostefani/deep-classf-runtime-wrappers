/*
==============================================================================*/
#include "RTNeural-classifier-lib.h"

#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <utility>
#include <cmath>
#include <numeric>
#include <limits> // std::numeric_limits
#include <vector>
#include <iomanip>
#include <sstream>

#include "RTNeural.h"

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
        RTNeural::DenseT<float, 350, 8>> model_t;

// Definition of the classifier class
class Classifier
{
public:
    /** Constructor */
    Classifier(const std::string &filename, bool verbose = false);
    /** Destructor */
    ~Classifier();
    /** Internal classification function, called by wrappers */
    int classify_internal(const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses);

private:
    /** Load the .onnx model and create inference session */
    model_t *loadModel(const std::string &filename, bool verbose=false);

    /** ind the index of the maximum value in an array */
    int argmax(const float vec[], size_t vecSize) const;

    //--------------------------------------------------------------------------
    model_t *model;

    size_t inputTensorSize = 173;
    size_t outputTensorSize = 8;
    std::vector<float> inputTensorValues;
};

Classifier::Classifier(const std::string &filename, bool verbose)
{
    // Load model
    if (verbose)
    {
        std::cout << std::setfill('-') << std::setw(40) << "" << std::endl;
        std::cout << "Parsing model..." << std::endl;
    }
    this->model = loadModel(filename, verbose);
    if (verbose)
    {
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
    delete this->model;
}

int Classifier::classify_internal(const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses)
{
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
    float max_out = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < numClasses; ++i)
    {
        outputVector[i] = this->model->getOutputs()[i];
        if (outputVector[i] > max_out)
            max_out = outputVector[i];
    }

    for(size_t i = 0; i < 8; i++)
    {
        std::cout << "Output_before_softmax[" << i << "] = " << outputVector[i] << std::endl;
    }

    // Normalize output for stable softmax
    for (size_t i = 0; i < numClasses; ++i)
        outputVector[i] -= max_out;

    for(size_t i = 0; i < 8; i++)
    {
        std::cout << "Output_normalized_for_softmax[" << i << "] = " << outputVector[i] << std::endl;
    }

    // Softmax
    float tsum = 0;
    for (size_t i = 0; i < numClasses; ++i)
        tsum += std::exp(outputVector[i]);
    std::cout << "tsum = " << tsum << std::endl;
    for (size_t i = 0; i < numClasses; ++i)
        outputVector[i] = std::exp(outputVector[i]) / tsum;

    for(size_t i = 0; i < 8; i++)
    {
        std::cout << "Output_after_softmax[" << i << "] = " << outputVector[i] << std::endl;
    }

    return argmax(outputVector, numClasses);
}

model_t *Classifier::loadModel(const std::string &filename, bool verbose)
{
    auto modelT = new model_t;
    std::ifstream jsonStream(filename, std::ifstream::binary);
    modelT->parseJson(jsonStream, verbose);
    return modelT;
}

int Classifier::argmax(const float vec[], size_t vecSize) const
{
    float max = std::numeric_limits<float>::lowest();
    int argmax = -1;
    for (size_t i = 0; i < vecSize; ++i)
    {
        if (vec[i] > max)
        {
            argmax = i;
            max = vec[i];
        }
    }
    return argmax;
}

/***** Handle functions *****/
ClassifierPtr createClassifier(const std::string &filename, bool verbose)
{
    return new Classifier(filename, verbose);
}

void deleteClassifier(ClassifierPtr cls)
{
    if (cls)
        delete cls;
}

int classify(ClassifierPtr cls, const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses)
{
    return cls->classify_internal(featureVector, numFeatures, outputVector, numClasses);
}