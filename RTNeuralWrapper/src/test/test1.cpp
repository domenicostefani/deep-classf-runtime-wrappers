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

#include "../rtneuralwrapper.h"

const std::size_t IN_SIZE = 173,
                  OUT_SIZE = 8;

void run_test(int n, ClassifierPtr &tc, std::array<float,IN_SIZE> &input, std::array<float,OUT_SIZE> &output)
{
    for(int i=0; i<n; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        int result = classify(tc,input,output);
        auto stop = std::chrono::high_resolution_clock::now();

        printf("Predicted class %d confidence: %f\n", result, output[result]);
        std::cout << "It took " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "us" << std::endl;
        std::cout << "(or " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms)" << std::endl;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "USAGE:\ntest-onnx-lib <onnx model path> \n");
        return 1;
    }
    const char* filename_cstr = argv[1];
    std::string filename(filename_cstr);

    ClassifierPtr tc = createClassifier(filename, true);

    // Output array
    std::array<float,OUT_SIZE> my_output_vec;
    std::array<float,IN_SIZE> my_input_vec;

    /* TEST 1
    *  Expected label is 3
    */
    std::cout << std::endl << "RUNNING TEST 1" << std::endl;
    my_input_vec = { 4.9166665e+00, 3.0283552e-01, 1.0394287e-01, 1.2451567e-01, 1.3391644e-01, 1.0785788e-01, 7.2398528e-02, 3.2941757e-03, 4.0917803e-02, 4.5008674e-02, 2.4652788e-02, 2.2933278e-02, 2.9465886e-02, 2.2034766e-02, 2.8299334e-02, 2.2654207e-02, 5.3271856e-03, 2.1723115e-03, 4.5960704e-03, 7.1194265e-03, 6.1773271e-03, 7.2930907e-03, 1.6396578e-02, 1.2050763e-02, 7.4279932e-03, 1.5410234e-02, 1.0335688e-02, 1.1184381e-02, 1.0974926e-02, 1.6968017e-02, 2.0399870e-02, 2.7399011e-02, 1.7038703e-02, 8.6462889e-03, 1.1360002e-02, 1.1510964e-02, 6.4039426e-03, 1.0874364e-02, 1.2017952e-02, 8.3324416e-03, 5.2536023e-03, 9.4272150e-03, 1.1449445e-02, 5.7383263e-01, 3.5592315e-01, 4.4857985e-01, 3.1027916e-01, 2.0711340e-01, 1.4867204e-01, 1.8857613e-01, 1.8207899e-01, 1.4608547e-01, 1.9119799e-01, 1.4240116e-01, 5.3622395e-02, 2.5337556e-02, 4.6004005e-02, 4.8264902e-02, -5.4831509e-03, 2.9759429e-02, 8.4432922e-03, -5.7074647e-02, -6.6851303e-02, -8.3122015e-02, -8.7512396e-02, -1.0607210e-01, -8.4255077e-02, -3.2448962e-02, -4.6821337e-02, -5.8725722e-02, -2.1231102e-02, -2.0720717e-02, 1.3980789e-02, 6.9901973e-02, 5.5653308e-02, 3.5454981e-02, 5.0483342e-02, 4.6005890e-02, 1.4778514e-02, -1.0923735e-02, -2.2807080e-02, -2.7634518e-02, -2.1296412e-02, -4.5426188e+00, 1.4651982e+00, 7.9958968e-02, 1.4270362e-01, 5.8920853e-02, 1.2307048e-01, 1.2142586e-01, -3.5220910e-02, -8.4764838e-02, 2.2346519e-01, 1.3543962e-02, 6.7931630e-02, 8.3043948e-03, 6.4206056e-02, 8.1757419e-02, 8.8886499e-02, 6.5111853e-02, 7.8697920e-02, -4.7458619e-02, 4.1117929e-02, -5.1116034e-02, 1.0735826e-01, 1.2188710e-02, 6.2339578e-02, 1.2425385e-02, -2.0671085e-02, 5.0440513e-02, 5.8763544e-03, 7.6302074e-02, 5.0762866e-02, 8.3499942e-03, -1.7459655e-02, -1.6574614e-02, -1.1408013e-02, -3.4473140e-02, -3.3456113e-02, -1.2982816e-02, 2.5486846e-03, -7.6258704e-03, 3.3074111e-02, 2.6510239e-02, -4.2178586e-02, 2.9806292e-03, -1.0117543e-02, -4.1564304e-02, -2.9643780e-02, -4.9470067e-02, 2.7098595e-03, 5.2331656e-02, -5.4460853e-02, 1.4533921e-02, 5.1873796e-02, -6.7446113e-02, 1.9630129e-02, 7.4984520e-03, 1.1324446e-02, -5.0794082e-03, 2.9561674e-02, 4.1259807e-02, 2.1030879e-02, 5.9662092e-01, 3.1572238e-01, 3.8431820e-01, 3.0795792e-01, 2.0698604e-01, 8.6641945e-02, 5.7130113e-02, 1.0229237e-01, 6.5894581e-02, 6.5611489e-02, 1.1805633e-01, 1.3625997e-01, 1.0723600e-01, 8.3205579e-03, -6.0935185e-04, 1.1614352e-02, -3.8095627e-02, -4.0525053e-02, -3.2665107e-02, -2.4945971e-02, -3.6342334e-02, -7.0326388e-02, -6.7431360e-02, -4.0976591e-02, -6.5504827e-02, -8.1277594e-02, -5.1742759e-02, -4.2006444e-02, -4.1926302e-02, -1.4157782e-02 };
    run_test(4, tc, my_input_vec, my_output_vec);

    deleteClassifier(tc);

    return 0;
}