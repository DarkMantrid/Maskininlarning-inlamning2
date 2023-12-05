#include <conv_layer.hpp>
#include <pooling_layer.hpp>

namespace ml = yrgo::machine_learning;

int main() {
    const std::vector<std::vector<double>> input{
        { 0, 1, 2, 4, 5 },
        { 6, 7, 8, 9, 10 },
        { -1, 0, 5, 2, 6 },
        { 10, 15, 2, 6, 8 },
        { 34, 3, 2, 5.6, 7 }
    };

    ml::ConvLayer conv1{5, 3}; // Creating a ConvLayer object with an image size of 5x5 and a kernel size of 3x3

    conv1.Feedforward(input);
    conv1.Print();  // Print the results or perform further operations
}
