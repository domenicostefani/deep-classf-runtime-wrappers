/*
==============================================================================*/
#include "tflitewrapper.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <limits>  // std::numeric_limits
#include <utility>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define LOG(x) std::cerr

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x)) {                                                  \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

// Definition of the classifier class
class Classifier {
public:
    /** Constructor */
    Classifier(const std::string &filename, bool verbose = false);
    /** Internal classification function, called by wrappers */
    int classify_internal(const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses, bool verbose = false);

private:
    /** Step 1, TFLITE loading the .tflite model */
    std::unique_ptr<tflite::FlatBufferModel> loadModel(const std::string &filename);
    /** Step 2, TFLITE building the interpreter */
    std::unique_ptr<Interpreter> buildInterpreter(const std::unique_ptr<tflite::FlatBufferModel> &model);

    /** ind the index of the maximum value in an array */
    int argmax(const float vec[], size_t vecSize) const;

    /** Check the input size requested by a tflite model */
    int requestedInputSize(const std::unique_ptr<Interpreter> &interpreter) const;
    /** Check the output size requested by a tflite model */
    int requestedOutputSize(const std::unique_ptr<Interpreter> &interpreter) const;

    //--------------------------------------------------------------------------

    std::unique_ptr<FlatBufferModel> model;
    std::unique_ptr<Interpreter> interpreter;

    float *inputTensorPtr, *outputTensorPtr;
};

Classifier::Classifier(const std::string &filename, bool verbose) {
    // Load model
    this->model = loadModel(filename);

    // Build the interpreter
    this->interpreter = buildInterpreter(model);
    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    assert(interpreter != nullptr);

    // Get pointer to the input Tensor
    this->inputTensorPtr = interpreter->typed_input_tensor<float>(0);
    // Get pointer to the output Tensor
    this->outputTensorPtr = interpreter->typed_output_tensor<float>(0);

    // Prime the classifier
    std::vector<float> pIv(requestedInputSize(interpreter));
    std::vector<float> pOv(requestedOutputSize(interpreter));
    this->classify_internal(&pIv[0], pIv.size(), &pOv[0], pOv.size(), verbose);

    /*
     * The priming operation should ensure that every allocation performed
     * by the Invoke method is perfomed here and not in the real-time thread.
     */
}

int Classifier::classify_internal(const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses, bool verbose) {
    size_t requestedInSize = requestedInputSize(interpreter);
    if (numFeatures != requestedInSize)
        throw std::logic_error("Error, input vector has to have size: " + std::to_string(requestedInSize) + " (Found " + std::to_string(numFeatures) + " instead)");

    // Fill `input`.
    for (int i = 0; i < numFeatures; ++i)
        this->inputTensorPtr[i] = featureVector[i];

    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    size_t requestedOutSize = requestedOutputSize(interpreter);
    if (numClasses != requestedOutSize)
        throw std::logic_error("Error, output vector has to have size: " + std::to_string(requestedOutSize) + " (Found " + std::to_string(numClasses) + " instead)");

    for (int i = 0; i < numClasses; ++i)
        outputVector[i] = outputTensorPtr[i];

    if (verbose)
        std::cout << "classify | Done." << std::endl
                  << std::flush;

    int res = argmax(outputVector, numClasses);

    if (verbose)
        std::cout << "classify | Done." << std::endl
                  << std::flush;
    if (verbose) {
        for (size_t i = 0; i < numClasses; ++i)
            std::cout << "classify | outputVector[" << i << "] :" << outputVector[i] << std::endl
                      << std::flush;
    }

    return res;
}

/** STEP 1 */
std::unique_ptr<tflite::FlatBufferModel> Classifier::loadModel(const std::string &filename) {
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(filename.c_str());
    TFLITE_MINIMAL_CHECK(model != nullptr);
    return model;
}

/** STEP 2 */
std::unique_ptr<Interpreter> Classifier::buildInterpreter(const std::unique_ptr<tflite::FlatBufferModel> &model) {
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    return interpreter;
}

int Classifier::argmax(const float vec[], size_t vecSize) const {
    float max = std::numeric_limits<float>::min();
    int argmax = -1;
    for (size_t i = 0; i < vecSize; ++i) {
        if (vec[i] > max) {
            argmax = i;
            max = vec[i];
        }
    }
    return argmax;
}

int Classifier::requestedInputSize(const std::unique_ptr<Interpreter> &interpreter) const {
    // get input dimension from the input tensor metadata
    // assuming one input only
    int input = interpreter->inputs()[0];
    TfLiteIntArray *dims = interpreter->tensor(input)->dims;

    int wanted_size = dims->data[1];
    return wanted_size;
}

int Classifier::requestedOutputSize(const std::unique_ptr<Interpreter> &interpreter) const {
    int output_index = interpreter->outputs()[0];
    TfLiteIntArray *output_dims = interpreter->tensor(output_index)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size = output_dims->data[output_dims->size - 1];
    return output_size;
}

/***** Handle functions *****/
ClassifierPtr createClassifier(const std::string &filename, bool verbose) {
    return new Classifier(filename, verbose);
}

void deleteClassifier(ClassifierPtr cls) {
    if (cls)
        delete cls;
}

int classify(ClassifierPtr cls, const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses, bool verbose) {
    return cls->classify_internal(featureVector, numFeatures, outputVector, numClasses, verbose);
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