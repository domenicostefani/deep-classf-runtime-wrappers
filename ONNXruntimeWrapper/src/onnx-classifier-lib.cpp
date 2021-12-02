/*
==============================================================================*/
#include "onnx-classifier-lib.h"

#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <utility>
#include <cmath>
#include <numeric>
#include <limits> // std::numeric_limits
#include <vector>

#include "onnxruntime_cxx_api.h"

/** Function to perform the product of the elements of a vector */
template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

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
    Ort::Session *loadModel(const std::string &filename);

    /** ind the index of the maximum value in an array */
    int argmax(const float vec[], size_t vecSize) const;

    //--------------------------------------------------------------------------
    Ort::Session *session;

    size_t inputTensorSize;
    size_t outputTensorSize;
    std::vector<float> inputTensorValues;
    std::vector<float> outputTensorValues;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
};

Classifier::Classifier(const std::string &filename, bool verbose)
{
    // Load model
    this->session = loadModel(filename);

    Ort::AllocatorWithDefaultOptions allocator;
    /*
    size_t numInputNodes = session->GetInputCount();
    size_t numOutputNodes = session->GetOutputCount();
    */

    const char* inputName = session->GetInputName(0, allocator);
    Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    const char* outputName = session->GetOutputName(0, allocator);
    Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();


    inputTensorSize = vectorProduct(inputDims);
    inputTensorValues = std::vector<float>(inputTensorSize);
    
    outputTensorSize = vectorProduct(outputDims);
    outputTensorValues = std::vector<float>(outputTensorSize);

    inputNames.push_back(inputName);
    outputNames.push_back(outputName);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));

    /*
    // Prime the classifier
    std::vector<float> pIv(inputTensorSize);
    std::vector<float> pOv(outputTensorSize);

    for(int j=0; j < 5; ++j)
        this->classify_internal(&pIv[0], pIv.size(), &pOv[0], pOv.size());
    */
    /*
     * The priming operation should ensure that every allocation performed
     * by the Invoke method is perfomed here and not in the real-time thread.
    */
}

Classifier::~Classifier() {
    delete this->session;
}

int Classifier::classify_internal(const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses)
{
    if (numFeatures != inputTensorSize)
        throw std::logic_error("Error, input vector has to have size: " + std::to_string(inputTensorSize) + " (Found " + std::to_string(numFeatures) + " instead)");

    // Fill `input`.
    for (size_t i = 0; i < numFeatures; ++i)
        inputTensorValues[i] = featureVector[i];

    // Run inference
    this->session->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);

    if (numClasses != outputTensorSize)
        throw std::logic_error("Error, output vector has to have size: " + std::to_string(outputTensorSize) + " (Found " + std::to_string(numClasses) + " instead)");

    // Copy output
    for (size_t i = 0; i < numClasses; ++i)
        outputVector[i] = outputTensorValues.at(i);

    // Softmax
    float tsum = 0;
    for (size_t i = 0; i < numClasses; ++i)
        tsum += std::exp(outputVector[i]);
    for (size_t i = 0; i < numClasses; ++i)
        outputVector[i] = std::exp(outputVector[i]) / tsum;

    return argmax(outputVector, numClasses);
}

Ort::Session *Classifier::loadModel(const std::string &filename)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-test");
    Ort::SessionOptions session_options;
    return new Ort::Session(env, filename.c_str(), session_options);
}

int Classifier::argmax(const float vec[], size_t vecSize) const
{
    float max = std::numeric_limits<float>::min();
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