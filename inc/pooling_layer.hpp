#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <conv_layer.hpp>

namespace yrgo {
namespace machine_learning {

class PoolingLayer : public ConvLayer {
  private:   
    std::vector<std::vector<double>> output_{};
    size_t stride_{1};

  public:
    enum class Type { kMax, kAverage };

    PoolingLayer(const size_t size = 2, const size_t stride = 1);

    bool Feedforward(const std::vector<std::vector<double>>& input,
                     const Type pooling_type = Type::kMax);

    void Print(std::ostream& ostream = std::cout, const int num_decimals = 1);
};

} /* namespace machine_learning */
} /* namespace yrgo */