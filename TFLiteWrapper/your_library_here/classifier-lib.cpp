/*
==============================================================================*/
#include "classifier-lib.hpp"

#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <utility>
#include <limits>       // std::numeric_limits
#include <chrono>  //TODO: remove

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define LOG(x) std::cerr

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
    if (!(x)) {                                                \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

class Classifier
{
public:
    Classifier(const std::string &filename, bool verbose = false);
    std::pair<int,float> classify(const std::vector<float> &featureVector, std::vector<float> &out_vec);

private:
    /** STEP 1 */
    std::unique_ptr<tflite::FlatBufferModel> loadModel(const std::string &filename);
    /** STEP 2 */
    std::unique_ptr<Interpreter> buildInterpreter(const std::unique_ptr<tflite::FlatBufferModel> &model);

    int maxClass(const std::vector<float> &vec) const;

    int requestedInputSize(const std::unique_ptr<Interpreter> &interpreter) const;
    int requestedOutputSize(const std::unique_ptr<Interpreter> &interpreter) const;

    //--------------------------------------------------------------------------
    std::unique_ptr<FlatBufferModel> model;
    std::unique_ptr<Interpreter> interpreter;

    float *inputTensorPtr, *outputTensorPtr;
};

Classifier::Classifier(const std::string &filename, bool verbose)
{
    // Load model
    this->model = loadModel(filename);

    // Build the interpreter
    this->interpreter = buildInterpreter(model); //TODO: move this?
    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    assert(interpreter != nullptr);

    // Get pointer to the input Tensor
    this->inputTensorPtr = interpreter->typed_input_tensor<float>(0);
    // Get pointer to the output Tensor
    this->outputTensorPtr = interpreter->typed_output_tensor<float>(0);

    // Prime the classifier
    std::vector<float> pIv(requestedInputSize(interpreter),0.0f);
    std::vector<float> pOv(requestedOutputSize(interpreter),0.0f);
    this->classify(pIv,pOv);
    /*
     * The priming operation SHOULD ensure that every allocation performed
     * by the Invoke method is perfomed here and not in the real-time thread.
     * 
     * I'm not experienced in thread-safe programming but running this in
     * Xenomai does not seem to cause a mode switch
    */
}

std::pair<int,float> Classifier::classify(const std::vector<float> &featureVector, std::vector<float>& out_vec)
{
    size_t requestedInSize = requestedInputSize(interpreter);
    if (featureVector.size() != requestedInSize)
        throw std::logic_error("Error, input vector has to have size: " + std::to_string(requestedInSize));

    // Fill `input`.
    for(int i=0; i<featureVector.size(); ++i)
        this->inputTensorPtr[i] = featureVector[i];

    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    size_t requestedOutSize = requestedOutputSize(interpreter);
    if(out_vec.size() != requestedOutSize)
        throw std::logic_error("Error, output vector has to have size: " + std::to_string(requestedOutSize));        

    for(int i=0; i<out_vec.size(); ++i)
        out_vec[i] = outputTensorPtr[i];

    // Softmax
    double tsum = 0;
    for(int i=0; i<out_vec.size(); ++i)
        tsum += exp(out_vec[i]);
    for(int i=0; i<out_vec.size(); ++i)
        out_vec[i] = exp(out_vec[i])/tsum;

    int maxclass = maxClass(out_vec);
    
    return std::make_pair(maxclass,out_vec[maxclass]);
}


/** STEP 1 */
std::unique_ptr<tflite::FlatBufferModel> Classifier::loadModel(const std::string &filename)
{
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(filename.c_str());
    TFLITE_MINIMAL_CHECK(model != nullptr);
    return model;
}

/** STEP 2 */
std::unique_ptr<Interpreter> Classifier::buildInterpreter(const std::unique_ptr<tflite::FlatBufferModel> & model)
{
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    return interpreter;
}

int Classifier::maxClass(const std::vector<float> &vec) const
{
    float max = std::numeric_limits<float>::min();
    int argmax = -1;
    for(int i = 0; i < vec.size(); ++i) {
        if(vec[i] > max) {
            argmax = i;
            max = vec[i];
        }
    }
    return argmax;
}

int Classifier::requestedInputSize(const std::unique_ptr<Interpreter>& interpreter) const
{
    // get input dimension from the input tensor metadata
    // assuming one input only
    int input = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;

    int wanted_size = dims->data[1];
    return wanted_size;
}

int Classifier::requestedOutputSize(const std::unique_ptr<Interpreter>& interpreter) const
{
    int output_index = interpreter->outputs()[0];
    TfLiteIntArray* output_dims = interpreter->tensor(output_index)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size = output_dims->data[output_dims->size - 1];
    return output_size;
}

/***** Handle functions *****/

ClassifierPtr createClassifier(const std::string &filename, bool verbose)
{
    ClassifierPtr cls = new Classifier(filename,verbose);
    return cls;
}

std::pair<int,float> classify(ClassifierPtr cls, const std::vector<float>& featureVector, std::vector<float>& outVector)
{
    std::pair<int,float> ret;
    if(cls)
        ret = cls->classify(featureVector, outVector);
    return ret;
}

void deleteClassifier(ClassifierPtr cls)
{
    if(cls)
        delete cls;
}