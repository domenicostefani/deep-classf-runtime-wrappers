/*
==============================================================================*/
#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <utility>
#include <limits>       // std::numeric_limits>
#include <array>
#include <chrono>  //TODO: remove
#include <cstdlib>

#include "classifier-lib.hpp"

const std::size_t IN_SIZE = 190,
                  OUT_SIZE = 8;

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "USAGE:\ninference_x86_64 <tflite model path> \n");
        return 1;
    }
    const char* filename_cstr = argv[1];
    std::string filename(filename_cstr);

    ClassifierPtr tc = createClassifier(filename);

    // std::vector<float> my_input_vec = {1.000000,0.001555,-0.342933,-0.006383,-0.236133,-0.013969,0.127167,0.074172,0.154741,-0.037491,-0.046996,0.040371,-0.074079,0.058866,0.081011,-0.059392,-0.078424,-0.009766,0.069354,0.024080,0.031791,0.000453,-0.087566,-0.002964,0.030200}; //TODO: read this from somewhere
    std::array<float,IN_SIZE> my_input_vec;
    for(int i=0; i<IN_SIZE; ++i)
        my_input_vec[i] = 1.0f;
    // std::vector<float> my_output_vec(4,0.0);
    std::array<float,OUT_SIZE> my_output_vec;
    for(int i=0; i<OUT_SIZE; ++i)
        my_output_vec[i] = 0.0f;

    for(int i=0; i<4; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        int result = classify(tc,my_input_vec,my_output_vec);
        auto stop = std::chrono::high_resolution_clock::now();

        printf("Predicted class %d confidence: %f\n", result, my_output_vec[result]);
        std::cout << "It took " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "us" << std::endl;
        std::cout << "(or " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms)" << std::endl;
    }
    deleteClassifier(tc);

    return 0;
}
