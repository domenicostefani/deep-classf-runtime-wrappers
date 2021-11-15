/*
==============================================================================*/
#include "classifier-lib.hpp"

#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <utility>
#include <limits> // std::numeric_limits
#include <vector>

#include "torch/script.h"

#define LOG(x) std::cerr

#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x))                                                    \
    {                                                            \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    } // TODO: check if needed

// Definition of the classifier class
class Classifier
{
public:
    /** Constructor */
    Classifier(const std::string &filename, bool verbose = false);
    /** Internal classification function, called by wrappers */
    int classify_internal(const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses);

private:
    /** Step 1, TORCHSCRIPT loading the .pt model */
    torch::jit::Module *loadModel(const std::string &filename);
    /** Step 2, TORCHSCRIPT run optimizations on given model */
    void prepareOptimize(torch::jit::Module *model);

    /** ind the index of the maximum value in an array */
    int argmax(const float vec[], size_t vecSize) const;

    /** Check the input size requested by a torchscript model */
    int requestedInputSize(const torch::jit::Module *model) const;
    /** Check the output size requested by the model */
    int requestedOutputSize(const torch::jit::Module *model) const;

    //--------------------------------------------------------------------------

    torch::jit::Module *model;

    std::vector<torch::jit::IValue> input_;
    float *input_data_;
    at::Tensor output_;
};

Classifier::Classifier(const std::string &filename, bool verbose)
{
    // Load model
    this->model = loadModel(filename);

    // Prepare model for inference and run optimizations
    prepareOptimize(this->model);

    // Initialize input Tensor
    this->input_.push_back(at::zeros({1, 180}));
    this->input_data_ = this->input_[0].toTensor().data_ptr<float>();
    // Initialize output Tensor
    this->output_ = at::zeros({8});

    // Prime the classifier
    std::vector<float> pIv(requestedInputSize(this->model));
    std::vector<float> pOv(requestedOutputSize(this->model));
    this->classify_internal(&pIv[0], pIv.size(), &pOv[0], pOv.size());

    /*
     * The priming operation should ensure that every allocation performed
     * by the Invoke method is perfomed here and not in the real-time thread.
    */
}

int Classifier::classify_internal(const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses)
{
    size_t requestedInSize = requestedInputSize(this->model);
    if (numFeatures != requestedInSize)
        throw std::logic_error("Error, input vector has to have size: " + std::to_string(requestedInSize) + " (Found " + std::to_string(numFeatures) + " instead)");

    // Fill `input`.
    for (int i = 0; i < numFeatures; ++i)
        this->input_data_[i] = featureVector[i];

    // Run inference
    this->output_ = this->model->forward(this->input_).toTensor();

    size_t requestedOutSize = requestedOutputSize(this->model);
    if (numClasses != requestedOutSize)
        throw std::logic_error("Error, output vector has to have size: " + std::to_string(requestedOutSize) + " (Found " + std::to_string(numClasses) + " instead)");

    outputVector = this->output_.data_ptr<float>();

    // Softmax
    double tsum = 0;
    for (int i = 0; i < numClasses; ++i)
        tsum += exp(outputVector[i]);
    for (int i = 0; i < numClasses; ++i)
        outputVector[i] = exp(outputVector[i]) / tsum;

    return argmax(this->output_.data_ptr<float>(), numClasses);
}

/** STEP 1 */
torch::jit::Module *Classifier::loadModel(const std::string &filename)
{
    torch::jit::Module module;
    try
    {
        module = torch::jit::load(filename);
    }
    catch (const c10::Error &e)
    {
        LOG("Error loading the model.\n");
        return false;
    }
    torch::jit::Module *model = new torch::jit::Module(module);
    return model;
}

/** STEP 2 */
void Classifier::prepareOptimize(torch::jit::Module *model)
{
    model->eval();
    *model = torch::jit::optimize_for_inference(*model);
}

int Classifier::argmax(const float vec[], size_t vecSize) const
{
    float max = std::numeric_limits<float>::min();
    int argmax = -1;
    for (int i = 0; i < vecSize; ++i)
    {
        if (vec[i] > max)
        {
            argmax = i;
            max = vec[i];
        }
    }
    return argmax;
}

int Classifier::requestedInputSize(const torch::jit::Module *model) const
{
    // get input dimension from the input tensor metadata
    // assuming one input only
    int wanted_size = (*model->parameters().begin()).size(1);
    return wanted_size;
}

int Classifier::requestedOutputSize(const torch::jit::Module *model) const // TODO: Check if optimizable
{
    auto iter = model->parameters().begin();
    for (size_t i = 0; i < model->parameters().size() - 1; i++)
        iter++;
    int output_size = (*iter).size(0);
    return output_size;
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
