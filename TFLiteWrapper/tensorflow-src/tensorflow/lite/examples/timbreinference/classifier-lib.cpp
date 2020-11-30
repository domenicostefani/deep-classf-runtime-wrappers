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
    std::pair<int,float> classify(std::vector<float> featureVector, bool verbose = false);

private:
    /** STEP 1 */
    std::unique_ptr<tflite::FlatBufferModel> loadModel(const std::string &filename);
    /** STEP 2 */
    std::unique_ptr<Interpreter> buildInterpreter(const std::unique_ptr<tflite::FlatBufferModel> & model);

    int maxClass(std::vector<float> vec) const;

    int requestedInputSize(const std::unique_ptr<Interpreter>& interpreter, bool verbose = false) const;

    //--------------------------------------------------------------------------
    std::unique_ptr<FlatBufferModel> model;
};

ClassifierPtr createClassifier(const std::string &filename, bool verbose)
{
    ClassifierPtr cls = new Classifier(filename,verbose);
    return cls;
}

std::pair<int,float> classify(ClassifierPtr cls, std::vector<float> featureVector, bool verbose)
{
    std::pair<int,float> ret;
    if(cls)
        ret = cls->classify(featureVector,verbose);
    return ret;
}

void deleteClassifier(ClassifierPtr cls)
{
    if(cls)
        delete cls;
}

Classifier::Classifier(const std::string &filename, bool verbose)
{
    // Load model
    this->model = loadModel(filename);
}

std::pair<int,float> Classifier::classify(std::vector<float> featureVector, bool verbose)
{
    // Build the interpreter
    std::unique_ptr<Interpreter> interpreter = buildInterpreter(model);
    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    if(verbose)
    {
        printf("=== Pre-invoke Interpreter State ===\n");
        tflite::PrintInterpreterState(interpreter.get());
    }

    assert(interpreter != nullptr);
    size_t requestedSize = requestedInputSize(interpreter);
    if (featureVector.size() != requestedSize){
        throw std::logic_error("Error, input vector has to have size: " + std::to_string(requestedSize));
    }

    // Fill input buffers
    float* in_ptr = interpreter->typed_input_tensor<float>(0);

    // Fill `input`.
    for(int i=0; i<featureVector.size(); ++i)
        in_ptr[i] = featureVector[i];

    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    if(verbose)
    {
        printf("\n\n=== Post-invoke Interpreter State ===\n");
        tflite::PrintInterpreterState(interpreter.get());
    }

    // Read output buffers
    float* out_ptr = interpreter->typed_output_tensor<float>(0);

    std::vector<float> out_vec(4);
    for(int i=0; i<4; ++i)
        out_vec[i] = out_ptr[i];

    std::vector<float> softmax_vec(4);
    float tsum = 0;
    for(int i=0; i<4; ++i)
        tsum += exp(out_vec[i]);
    for(int i=0; i<4; ++i)
        softmax_vec[i] = exp(out_vec[i])/tsum;

    int maxclass = maxClass(out_vec);
    return std::make_pair(maxclass,softmax_vec[maxclass]);
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

int Classifier::maxClass(std::vector<float> vec) const
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

int Classifier::requestedInputSize(const std::unique_ptr<Interpreter>& interpreter, bool verbose) const
{
    // get input dimension from the input tensor metadata
    // assuming one input only
    int input = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;

    int wanted_size = dims->data[1];
    if(verbose)
        printf("Inputs size requested: %d\n", wanted_size);
    return wanted_size;
}
