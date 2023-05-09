/*
==============================================================================*/
#include "tflitewrapper.h"

#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <utility>
#include <limits>       // std::numeric_limits

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

// Definition of the classifier class
class Classifier
{
public:
    /** Constructor */
    Classifier(const std::string &filename, bool verbose = false);
    /** Internal classification function, called by wrappers */
    int classify_internal(const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses, bool verbose = false);
    int classify_internal2d(float **featureMatrix, size_t rows, size_t cols, float outputVector[], size_t numClasses, bool verbose = false);

    // int requestedInputSize(const std::unique_ptr<Interpreter> &interpreter) const;
    // int requestedOutputSize(const std::unique_ptr<Interpreter> &interpreter) const;

    int requestedInputSize() const;
    int requested2drows() const;
    int requested2dcols() const;
    int requestedOutputSize() const;
private:
    /** Step 1, TFLITE loading the .tflite model */
    std::unique_ptr<tflite::FlatBufferModel> loadModel(const std::string &filename);
    /** Step 2, TFLITE building the interpreter */
    std::unique_ptr<Interpreter> buildInterpreter(const std::unique_ptr<tflite::FlatBufferModel> &model);

    /** ind the index of the maximum value in an array */
    int argmax(const float vec[], size_t vecSize) const;

    /** Check the input size requested by a tflite model */

    //--------------------------------------------------------------------------

    std::unique_ptr<FlatBufferModel> model;
    std::unique_ptr<Interpreter> interpreter;

    float *inputTensorPtr, *outputTensorPtr;
};

Classifier::Classifier(const std::string &filename, bool verbose)
{
    // Load model
    if (verbose)
        std::cout << "Loading model: '" << filename << "'..." << std::endl;
    this->model = loadModel(filename);

    // Build the interpreter
    if (verbose)
        std::cout << "Done.\nBuilding interpreter..." << std::endl;
    this->interpreter = buildInterpreter(model);
    // Allocate tensor buffers.
    if (verbose)
        std::cout << "Done.\nAllocating tensor buffers..." << std::endl;
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    // assert(interpreter != nullptr);
    // Configure the interpreter
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(1);

    if (interpreter == nullptr)
        throw std::runtime_error("Failed to build interpreter. Return value is NULL.");

    if (verbose) {
        std::cout << "Interpreter built successfully." << std::endl;
        tflite::PrintInterpreterState(interpreter.get()); 
    }


    // Get pointer to the input Tensor
    if (verbose)
        std::cout << "Done.\nGetting pointer to the input tensor..." << std::endl;
    if (interpreter->inputs().size() != 1)
        throw std::runtime_error("Error, the model has more than one input tensor.  This is not supported.");
    else if (verbose) {
        std::cout << "The model has " << interpreter->inputs().size() << " input tensors." << std::endl;
        for (size_t i=0; i<interpreter->inputs().size(); ++i) {
            TfLiteIntArray* input_dims = interpreter->tensor(interpreter->inputs()[i])->dims;
            // Print all sizes of the input tensor
            for (int j=0; j<input_dims->size; ++j) {
                std::cout << "Input tensor [" << i << "] at interpreter index " << this->interpreter->inputs()[i] << " has dimension " << j << " with size: " << input_dims->data[j] << std::endl << std::flush;
            }
            // auto input_size = input_dims->data[input_dims->size - 1];
            // auto input_type = TfLiteTypeGetName(interpreter->tensor(interpreter->inputs()[i])->type);
            // std::cout << "Input tensor [" << i << "] at interpreter index " << this->interpreter->inputs()[i] << " has lenth: " << input_size << " and type: " << input_type << std::endl << std::flush;
        }
    }
    this->inputTensorPtr = interpreter->typed_input_tensor<float>(0); // Based on the code at https://github.com/google-coral/edgetpu/blob/75e675633c2110a991426c8afa64f122b16ac372/src/cpp/examples/model_utils.cc , the index used here always start from 0 and has no relation to the this->interpreter->inputs()[i]
    // assert (inputTensorPtr != nullptr);
    if (inputTensorPtr == nullptr)
        throw std::runtime_error("Failed to get pointer to the input tensor.  interpreter->typed_input_tensor<float>(0) returns NULL.");

    // Get pointer to the output Tensor
    if (verbose)
        std::cout << "Done.\nGetting pointer to the output tensor..." << std::endl;
    if (interpreter->outputs().size() != 1)
        throw std::runtime_error("Error, the model has more than one output tensor.  This is not supported.");
    else if (verbose) {
        std::cout << "The model has " << interpreter->outputs().size() << " output tensors." << std::endl;
        for (size_t i=0; i<interpreter->outputs().size(); ++i) {
            TfLiteIntArray* output_dims = interpreter->tensor(interpreter->outputs()[i])->dims;
            auto output_size = output_dims->data[output_dims->size - 1];
            auto output_type = TfLiteTypeGetName(interpreter->tensor(interpreter->outputs()[i])->type);
            std::cout << "Output tensor [" << i << "] at interpreter index " << this->interpreter->outputs()[i] << " has lenth: " << output_size << " and type: " << output_type << std::endl << std::flush;
        }
    }
    this->outputTensorPtr = interpreter->typed_output_tensor<float>(0); // Based on the code at https://github.com/google-coral/edgetpu/blob/75e675633c2110a991426c8afa64f122b16ac372/src/cpp/examples/model_utils.cc , the index used here always start from 0 and has no relation to the this->interpreter->outputs()[i]
    // assert (outputTensorPtr != nullptr);
    if (outputTensorPtr == nullptr)
        throw std::runtime_error("Failed to get pointer to the output tensor.  interpreter->typed_output_tensor<float>(0) returns NULL.");

    bool prime2d = (interpreter->tensor(interpreter->inputs()[0])->dims->size == 4);
    if (verbose) {
        std::cout << "prime2d: " << prime2d << std::endl;
        std::cout << "interpreter->tensor(interpreter->inputs()[0])->dims->size: " << interpreter->tensor(interpreter->inputs()[0])->dims->size << std::endl;
        if (prime2d)
            std::cout << "The model is a 2D model." << std::endl;
        else
            std::cout << "The model is a 1D model." << std::endl;
    }
    // Prime the classifier
    if (verbose)
        std::cout << "Done.\nPriming the classifier (Calling inference once)..." << std::endl;
    std::vector<float> pIv;
    if (prime2d) {
        if (verbose) {
            std::cout << "Input size: [" << this->requested2drows() << " x " << this->requested2dcols() << "] | Output size: " << this->requestedOutputSize() << std::endl;
        }
        pIv.resize(this->requested2drows()*this->requested2dcols());
    }
    else {
        if (verbose) {
            std::cout << "Input size: " << this->requestedInputSize()  << " | Output size: " << this->requestedOutputSize() << std::endl;
        }
        pIv.resize(this->requestedInputSize());
    }
    std::vector<float> pOv;
    pOv.resize(this->requestedOutputSize());
    this->classify_internal(&pIv[0],pIv.size(),&pOv[0],pOv.size(), verbose);
    if (verbose)
        std::cout << "Done.\nClassifier primed." << std::endl;

    /*
     * The priming operation should ensure that every allocation performed
     * by the Invoke method is perfomed here and not in the real-time thread.
    */
}

int Classifier::classify_internal(const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses, bool verbose)
{
    if (verbose) {
        std::cout << "classify | Input size: " << numFeatures  << " | Output size: " << numClasses << std::endl;
        std::cout << "classify | Filling input tensor..." << std::endl << std::flush;
    }
    // Fill `input`.
    for(size_t i=0; i<numFeatures; ++i) {
        this->inputTensorPtr[i] = featureVector[i];
        // if (verbose) {
        //     std::cout << "classify | inputTensorPtr[" << i << "] = " << this->inputTensorPtr[i] << std::endl << std::flush;
        // }
    }
     
    // Alternative with memcpy
    // memcpy(this->inputTensorPtr, featureVector, numFeatures*sizeof(float));

    if (verbose)
        std::cout << "classify | Done.\nclassify | Running inference..." << std::endl << std::flush;

    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    if (verbose)
        std::cout << "classify | Done.\nclassify | Reading output tensor..." << std::endl << std::flush;

    size_t requestedOutSize = requestedOutputSize();
    if(numClasses != requestedOutSize)
        throw std::logic_error("Error, output vector has to have size: " + std::to_string(requestedOutSize) + " (Found " +std::to_string(numClasses)+ " instead)");        

    if (verbose) {
        std::cout << "classify | Done (size is OK).\nclassify | Copying to array..." << std::endl << std::flush;

        for (size_t i=0; i<numClasses; ++i)
            std::cout << "classify | outputTensorPtr[" << i << "] :" << outputTensorPtr[i] << std::endl << std::flush;
    }
    for (size_t i=0; i<numClasses; ++i)
        outputVector[i] = outputTensorPtr[i];

    if (verbose)
        std::cout << "classify | Done.\nclassify | Applying softmax..." << std::endl << std::flush;

    // Softmax
    double tsum = 0;
    for (size_t i=0; i<numClasses; ++i)
        tsum += exp(outputVector[i]);
    for (size_t i=0; i<numClasses; ++i)
        outputVector[i] = exp(outputVector[i])/tsum;

    int res = argmax(outputVector, numClasses);

    if (verbose)
        std::cout << "classify | Done." << std::endl << std::flush;
    if (verbose) {
        for (size_t i=0; i<numClasses; ++i)
            std::cout << "classify | outputVector[" << i << "] :" << outputVector[i] << std::endl << std::flush;
    }


    return res;
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

int Classifier::argmax(const float vec[], size_t vecSize) const
{
    float max = std::numeric_limits<float>::min();
    int argmax = -1;
    for(size_t i = 0; i < vecSize; ++i) {
        if(vec[i] > max) {
            argmax = i;
            max = vec[i];
        }
    }
    return argmax;
}

int Classifier::requestedInputSize() const
{
    // get input dimension from the input tensor metadata
    // assuming one input only
    int input = this->interpreter->inputs()[0];
    TfLiteIntArray* dims = this->interpreter->tensor(input)->dims;

    int wanted_size = dims->data[1];
    return wanted_size;
}

int Classifier::requested2drows() const
{
    int input = this->interpreter->inputs()[0];
    return this->interpreter->tensor(input)->dims->data[1];
}

int Classifier::requested2dcols() const
{
    int input = this->interpreter->inputs()[0];
    return this->interpreter->tensor(input)->dims->data[2];
}


int Classifier::requestedOutputSize() const
{
    int output_index = this->interpreter->outputs()[0];
    TfLiteIntArray* output_dims = this->interpreter->tensor(output_index)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size = output_dims->data[output_dims->size - 1];
    return output_size;
}


/***** Handle functions *****/
ClassifierPtr createClassifier (const std::string &filename, bool verbose)
{
    ClassifierPtr res = new Classifier(filename, verbose);
    return res;
}

void deleteClassifier(ClassifierPtr cls)
{
    if(cls)
        delete cls;
}

int classify(ClassifierPtr cls, const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses, bool verbose)
{
    size_t requestedInSize = cls->requestedInputSize();
    if (numFeatures != requestedInSize)
        throw std::logic_error("Error, input vector has to have size: " + std::to_string(requestedInSize) + " (Found " +std::to_string(numFeatures)+ " instead)");
    
    return cls->classify_internal(featureVector, numFeatures, outputVector, numClasses, verbose);
}


int classifyFlat2D(ClassifierPtr cls, const float flatFeatureMatrix[], size_t nRows, size_t nCols, float outputVector[], size_t numClasses, bool verbose)
{
    size_t reqRows, reqCols;
    reqRows = cls->requested2drows();
    reqCols = cls->requested2dcols();
    if (nRows != reqRows || nCols != reqCols)
        throw std::logic_error("Error, input vector has to have size: " + std::to_string(reqRows) + "x" + std::to_string(reqCols) + " (Found " +std::to_string(nRows)+ "x" + std::to_string(nCols) + " instead)");
    return cls->classify_internal(flatFeatureMatrix, nRows*nCols, outputVector, numClasses, verbose);
}

int classify(ClassifierPtr cls, std::vector<float>& featureVector, std::vector<float>& outputVector)
{
    if (featureVector.size() != getModelInputSize1d(cls)) {
        std::cerr << "Input vector size does not match model input size (" << featureVector.size() << " != " << getModelInputSize1d(cls) << ")" << std::endl;
        throw std::runtime_error(("Input vector size does not match model input size (" + std::to_string(featureVector.size()) + " != " + std::to_string(getModelInputSize1d(cls)) + ")").c_str());
    }
    if (outputVector.size() != getModelOutputSize(cls)) {
        std::cerr << "Output vector size does not match model output size (" << outputVector.size() << " != " << getModelOutputSize(cls) << ")" << std::endl;
        throw std::runtime_error(("Output vector size does not match model output size (" + std::to_string(outputVector.size()) + " != " + std::to_string(getModelOutputSize(cls)) + ")").c_str());
    }
    return classify(cls, featureVector.data(), (size_t)featureVector.size(), outputVector.data(), (size_t)outputVector.size());
}

int classifyFlat2D(ClassifierPtr cls, std::vector<float>& flatInputMatrix, size_t nRows, size_t nCols, std::vector<float>& outputVector, bool verbose) {
    if (flatInputMatrix.size() != nRows*nCols) {
        std::cerr << "Input vector size does not match the nRows and nCols values provided (" << flatInputMatrix.size() << " != " << nRows << "*" << nCols << ")" << std::endl;
        throw std::runtime_error(("Input vector size does not match the nRows and nCols values provided (" + std::to_string(flatInputMatrix.size()) + " != " + std::to_string(nRows) + "*" + std::to_string(nCols) + ")").c_str());
    }
    if ((int)nRows != cls->requested2drows())
        throw std::runtime_error(("Input vector size does not match model input size (" + std::to_string(nRows) + " != " + std::to_string(cls->requested2drows()) + ")").c_str());
    if ((int)nCols != cls->requested2dcols())
        throw std::runtime_error(("Input vector size does not match model input size (" + std::to_string(nCols) + " != " + std::to_string(cls->requested2dcols()) + ")").c_str());
    if (outputVector.size() != getModelOutputSize(cls)) {
        std::cerr << "Output vector size does not match model output size (" << outputVector.size() << " != " << getModelOutputSize(cls) << ")" << std::endl;
        throw std::runtime_error(("Output vector size does not match model output size (" + std::to_string(outputVector.size()) + " != " + std::to_string(getModelOutputSize(cls)) + ")").c_str());
    }

    return classifyFlat2D(cls, flatInputMatrix.data(), nRows, nCols, outputVector.data(), outputVector.size(), verbose);
}



size_t getModelInputSize1d(ClassifierPtr cls) {
    return (size_t)(cls->requestedInputSize());
}

void getModelInputSize2d(ClassifierPtr cls, size_t& rows, size_t& columns) {
    rows = (size_t)(cls->requested2drows());
    columns = (size_t)(cls->requested2dcols());
}

size_t getModelOutputSize(ClassifierPtr cls) {
    return (size_t)(cls->requestedOutputSize());
}