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

const bool VERBOSE_CREATE = true;
const bool VERBOSE_CLASSIFY = true;

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        const char* scriptn_cstr = argv[0];
        std::string scriptn(scriptn_cstr);
        std::string errmessage = "USAGE:\n"+scriptn+" <model path> \n";
        fprintf(stderr, "%s", errmessage.c_str());
        return 1;
    }
    const char* filename_cstr = argv[1];
    std::string filename(filename_cstr);

    InferenceEngine::InterpreterPtr tc = InferenceEngine::createInterpreter(filename, VERBOSE_CREATE);

    size_t in_size = InferenceEngine::getModelInputSize1d(tc);
    size_t out_size = InferenceEngine::getModelOutputSize(tc);

    std::cout << "Model input size: " << in_size << std::endl;
    std::cout << "Model output size: " << out_size << std::endl;

    std::vector<float> my_input_vec;
    my_input_vec.resize(in_size);
    std::vector<float> my_output_vec;
    my_output_vec.resize(out_size);


    for(size_t i=0; i<in_size; ++i)
        my_input_vec[i] = 1.0f;
    for(size_t i=0; i<out_size; ++i)
        my_output_vec[i] = 0.0f;

    for(size_t i=0; i<4; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        InferenceEngine::invoke(tc,my_input_vec,my_output_vec);
        int result = std::distance(my_output_vec.begin(), std::max_element(my_output_vec.begin(), my_output_vec.end()));

        // int result = classify(tc,my_input_vec,my_output_vec,VERBOSE_CLASSIFY);
        // int result = thisisatest();
        auto stop = std::chrono::high_resolution_clock::now();

        // Print output vector
        for(size_t i=0; i<out_size; ++i)
            printf("Output vector %zu: %f\n", i, my_output_vec[i]);

        printf("Predicted class %d confidence: %f\n", result, my_output_vec[result]);
        std::cout << "It took " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "us" << std::endl;
        std::cout << "(or " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms)" << std::endl;
    }
    InferenceEngine::deleteInterpreter(tc);

    std::cout << std::endl << std::endl;
    std::cout << "#----------------------------------------------------#" << std::endl;
    std::cout << "# Test completed successfully                        #" << std::endl;
    std::cout << "#----------------------------------------------------#" << std::endl << std::endl;

    return 0;
}
