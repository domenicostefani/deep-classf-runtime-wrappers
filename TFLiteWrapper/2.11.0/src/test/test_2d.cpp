/*
==============================================================================*/
#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <utility>
#include <limits>
#include <array>
#include <chrono>
#include <cstdlib>

#include "../tflitewrapper.h"

// const bool verbose = false;
// const bool verbose = false;
const size_t N_REPEATED_TESTS = 5;

int main(int argc, char* argv[])
{
    srand(time(NULL));
    bool verbose = false;
    if (argc < 2 || argc > 3)
    {
        const char* scriptn_cstr = argv[0];
        std::string scriptn(scriptn_cstr);
        std::string errmessage = "USAGE:\n"+scriptn+" <model path> [-v]\n";
        fprintf(stderr, "%s", errmessage.c_str());
        return 1;
    } else if (argc == 3 && std::string(argv[2]) == "-v") {
        std::cout << "test2d | Verbose mode enabled" << std::endl;
        verbose = true;
    }

    const char* filename_cstr = argv[1];
    std::string filename(filename_cstr);

    InferenceEngine::InterpreterPtr tc = InferenceEngine::createInterpreter(filename, verbose);

    size_t in_rows, in_cols;
    InferenceEngine::getModelInputSize2d(tc, in_rows, in_cols);
    size_t out_size = InferenceEngine::getModelOutputSize(tc);

    std::cout << "test2d | Model input size: [" << in_rows <<  "," << in_cols << "]" << std::endl;
    std::cout << "test2d | Model output size: " << out_size << std::endl;

    // std::vector<std::vector<float>> my_input_vec;
    // my_input_vec.resize(in_rows);
    // for (size_t i = 0; i < in_rows; ++i)
    //     my_input_vec[i].resize(in_cols);

    float **my_input_matrix = new float*[in_rows];
    for (size_t i = 0; i < in_rows; ++i) {
        my_input_matrix[i] = new float[in_cols];
        for (size_t j = 0; j < in_cols; ++j) {
            my_input_matrix[i][j] = (float)rand() / RAND_MAX;
        }
    }

    // Transform matrix to 1d array
    std::vector<float> my_input_vec;
    my_input_vec.resize(in_rows*in_cols);
    for (size_t i = 0; i < in_rows; ++i)
        for (size_t j = 0; j < in_cols; ++j)
            my_input_vec[i*in_cols + j] = my_input_matrix[i][j];


    std::vector<float> my_output_vec;
    my_output_vec.resize(out_size);

    for(size_t i=0; i<out_size; ++i)
        my_output_vec[i] = 0.0f;

    for(size_t i=0; i<N_REPEATED_TESTS; ++i)
    {
        // Repopulate input vector
        for (size_t j=0; j<in_rows*in_cols; ++j)
            my_input_vec[j] = (float)rand() / RAND_MAX;

        auto start = std::chrono::high_resolution_clock::now();

        InferenceEngine::invokeFlat2D(tc, my_input_vec, in_rows, in_cols, my_output_vec, verbose);

        // int result = argmax(my_output_vec);
        // Find argmax with std::max_element
        int result = std::distance(my_output_vec.begin(), std::max_element(my_output_vec.begin(), my_output_vec.end()));

        auto stop = std::chrono::high_resolution_clock::now();

        printf("test2d | Predicted class %d confidence: %f\n", result, my_output_vec[result]);
        std::cout << "test2d | It took " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "us" << std::endl;
        if (std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() > 1000)
            std::cout << "test2d | (or " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms)" << std::endl;
    }
    InferenceEngine::deleteInterpreter(tc);

    std::cout << "test2d | Test completed successfully                        #" << std::endl;
    return 0;
}
