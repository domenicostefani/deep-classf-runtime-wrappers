/*
==============================================================================*/
#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <utility>
#include <limits>       // std::numeric_limits>
#include <vector>
#include <chrono>  //TODO: remove

#include "liteclassifier.h"

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "USAGE:\ninference_x86_64 <tflite model path>\n");
        return 1;
    }
    const char* filename_cstr = argv[1];
    std::string filename(filename_cstr);

    ClassifierPtr tc = createClassifier(filename);

    std::vector<float> my_input_vec = {1.000000,0.001555,-0.342933,-0.006383,-0.236133,-0.013969,0.127167,0.074172,0.154741,-0.037491,-0.046996,0.040371,-0.074079,0.058866,0.081011,-0.059392,-0.078424,-0.009766,0.069354,0.024080,0.031791,0.000453,-0.087566,-0.002964,0.030200}; //TODO: read this from somewhere
    std::vector<float> my_output_vec(4,0.0);


    auto start = std::chrono::high_resolution_clock::now();
    std::pair<int,float> result = classify(tc,my_input_vec,my_output_vec);
    auto stop = std::chrono::high_resolution_clock::now();

    printf("Predicted class %d confidence: %f\n", result.first, result.second);
    std::cout << "It took " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "us" << std::endl;
    std::cout << "(or " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms)" << std::endl;

    deleteClassifier(tc);

    return 0;
}
