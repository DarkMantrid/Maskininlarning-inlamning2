#pragma once

#include <iostream>
#include <iomanip>
#include <vector>

namespace yrgo {
namespace machine_learning {

class ConvLayer {
  protected:
    std::vector<std::vector<double>> image_{};
    std::vector<std::vector<double>> kernel_{};
    std::vector<std::vector<double>> output_{};
    double kernel_bias_{};

  public:
    ConvLayer(void) = default;

    ConvLayer(const size_t image_size, const size_t kernel_size);

    const std::vector<std::vector<double>>& Image(void) const { return image_; }

    const std::vector<std::vector<double>>& Kernel(void) const { return kernel_; }

    const std::vector<std::vector<double>>& Output(void) const { return output_; }

    double KernelBias(void) const { return kernel_bias_; }

    bool Feedforward(const std::vector<std::vector<double>>& input);

    static void PrintMatrix(const std::vector<std::vector<double>>& data,
                            std::ostream& ostream = std::cout,
                            const int num_decimals = 1,
                            const size_t offset = 0);

    void Print(std::ostream& ostream = std::cout, const int num_decimals = 1);

    void PerformConvolution(); 

    void ReLUActivation(); 
};

} /* namespace machine_learning */
} /* namespace yrgo */
