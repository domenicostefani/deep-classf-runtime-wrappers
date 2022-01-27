/*
==============================================================================*/
#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <utility>
#include <limits> // std::numeric_limits>
#include <array>
#include <chrono> //TODO: remove
#include <cstdlib>

#include "../tflitewrapper.h"

const std::size_t IN_SIZE = 180,
                  OUT_SIZE = 8;

#include <iostream>
#include <fstream>
#include <istream>
#include <string>
#include <vector>
#include <array>

enum class CSVState
{
    UnquotedField,
    QuotedField,
    QuotedQuote
};

std::vector<std::string> readCSVRow(const std::string &row)
{
    CSVState state = CSVState::UnquotedField;
    std::vector<std::string> fields{""};
    size_t i = 0; // index of the current field
    for (char c : row)
    {
        switch (state)
        {
        case CSVState::UnquotedField:
            switch (c)
            {
            case ',': // end of field
                fields.push_back("");
                i++;
                break;
            case '"':
                state = CSVState::QuotedField;
                break;
            default:
                fields[i].push_back(c);
                break;
            }
            break;
        case CSVState::QuotedField:
            switch (c)
            {
            case '"':
                state = CSVState::QuotedQuote;
                break;
            default:
                fields[i].push_back(c);
                break;
            }
            break;
        case CSVState::QuotedQuote:
            switch (c)
            {
            case ',': // , after closing quote
                fields.push_back("");
                i++;
                state = CSVState::UnquotedField;
                break;
            case '"': // "" -> "
                fields[i].push_back('"');
                state = CSVState::QuotedField;
                break;
            default: // end of quote
                state = CSVState::UnquotedField;
                break;
            }
            break;
        }
    }
    return fields;
}

/// Read CSV file, Excel dialect. Accept "quoted fields ""with quotes"""
std::vector<std::vector<std::string>> readCSV(std::istream &in, bool skipHeader = true)
{
    std::vector<std::vector<std::string>> table;
    std::string row;
    if(!in.eof() && skipHeader)
        std::getline(in, row);
    while (!in.eof())
    {
        std::getline(in, row);
        if (in.bad() || in.fail())
        {
            break;
        }
        auto fields = readCSVRow(row);
        table.push_back(fields);
    }
    return table;
}

std::vector<std::array<float,IN_SIZE>> stringVec2FeatureVec(const std::vector<std::vector<std::string>> &in)
{
    std::vector<std::array<float,IN_SIZE>> res;

    for (const auto row : in)
    {
        std::array<float,IN_SIZE> parz = std::array<float,IN_SIZE>();
        if (row.size() != IN_SIZE)
            throw std::logic_error("Wrong number of features in input file (found " + std::to_string(row.size()) + "instead of" + std::to_string(IN_SIZE) + ")");
        for (size_t idx = 0; idx < IN_SIZE; ++idx)
        {
            try
            {
                parz[idx] = std::stof(row[idx]);
            }
            catch(const std::invalid_argument& e)
            {
                std::cerr << e.what() << '\n' << "Failed converting '" << row[idx] << "' to float";
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }
            
        }
        res.push_back(parz);
    }

    return res;
}


std::vector<int> stringVec2LabelsVec(const std::vector<std::vector<std::string>> &in)
{
    std::vector<int>  res;
    for (const auto row : in)
    {
        if (row.size() != 1)
            throw std::logic_error("Wrong number of labels in a row of the input file (found " + std::to_string(row.size()) + "instead of" + std::to_string(IN_SIZE) + ")");

        // convert first and only string to float
        try {
            res.push_back(std::stof(row[0]));
        }
        catch(const std::invalid_argument& e) {
            std::cerr << e.what() << '\n' << "Failed converting '" << row[0] << "' to float";
        }
        catch(const std::exception& e) {
            std::cerr << e.what() << '\n';
        }
    }
    return res;
}

void printVec(std::vector<std::array<float,IN_SIZE>> &in, size_t row)
{
    if (row <= in.size()) {
        for (const auto col : in[row]) {
            std::cout << "'" << col << "' ";
        }
        std::cout << std::endl;
    }
}

void printVec(std::vector<std::array<float,IN_SIZE>> &in)
{
    for (int row = 0; row < in.size(); ++row)
        printVec(in, row);
}

int main(int argc, char *argv[])
{
    // Parse script arguments
    if (argc != 4)
    {
        const char *execpath_cstr = argv[0];
        std::string execpath(execpath_cstr);
        std::string errormsg = "USAGE:\n"+execpath+" <tflite model path> <features_file> <true_labels_file>\n";
        fprintf(stderr, errormsg.c_str());
        return 1;
    }
    const char *modelpath_cstr = argv[1];
    std::string modelpath(modelpath_cstr);

    const char *featurespath_cstr = argv[2];
    std::string featurespath(featurespath_cstr);

    const char *labelspath_cstr = argv[3];
    std::string labelspath(labelspath_cstr);

    // Read features file

    std::string line;
    std::ifstream featuresFile(featurespath);
    std::vector<std::array<float,IN_SIZE>> featureVectors = std::vector<std::array<float,IN_SIZE>>();
    if (featuresFile.is_open())
    {
        featureVectors = stringVec2FeatureVec(readCSV(featuresFile));
        // printVec(featureVectors,0); // for debugging
        featuresFile.close();
    }
    else
        throw std::logic_error("Unable to open file");

    // Read labels file

    std::ifstream labelsFile(labelspath);
    std::vector<int> y_true = std::vector<int>();
    if (labelsFile.is_open())
    {
        y_true = stringVec2LabelsVec(readCSV(labelsFile));
        labelsFile.close();
    }
    else
        throw std::logic_error("Unable to open file");



    ClassifierPtr tc = createClassifier(modelpath);

    std::array<float, OUT_SIZE> my_output_vec;
    for (int i = 0; i < OUT_SIZE; ++i)
        my_output_vec[i] = 0.0f;

    std::vector<int> y_pred = std::vector<int>();
    for (int i = 0; i < featureVectors.size(); ++i)
    {
        int result = classify(tc, featureVectors[i], my_output_vec);

        // printf("Predicted class %d confidence: %f\n", result, my_output_vec[result]);
        y_pred.push_back(result);
    }
    deleteClassifier(tc);



    std::cout << "Total feature vectors: " << featureVectors.size() << std::endl;
    if (y_pred.size() != y_true.size())
        throw std::logic_error("Number of predictions differs from the number of true labels (" + std::to_string(y_pred.size()) + "!=" + std::to_string(y_true.size()) + ")");


    auto cmatrix = std::array<std::array<int,OUT_SIZE>,OUT_SIZE>();


    for (int idx=0;idx<y_true.size();++idx) {
        int curTrue = y_true[idx], curPred = y_pred[idx];

        cmatrix[curTrue][curPred] += 1;
    }

    
    for (const auto &row : cmatrix) {
        std::cout << "[ ";

        for (const auto &val : row) {
            std::cout << val << " ";
        }

        std::cout << "]\n";

    }


    std::cout << std::endl << std::endl;
    std::cout << "#----------------------------------------------------#" << std::endl;
    std::cout << "# Test completed successfully                        #" << std::endl;
    std::cout << "#----------------------------------------------------#" << std::endl << std::endl;


    return 0;
}
