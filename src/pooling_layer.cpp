#include <pooling_layer.hpp>

namespace yrgo {
namespace machine_learning {

PoolingLayer::PoolingLayer(const size_t size, const size_t stride) {

}

bool PoolingLayer::Feedforward(const std::vector<std::vector<double>>& input, const Type pooling_type)
{
    if (input.size() < output_.size()) {
        std::cerr << "Input image cannot be smaller than pooling layer size!\n\n";
        return false;
    } else {
        return true;
    }
}

void PoolingLayer::Print(std::ostream& ostream, const int num_decimals) {
    if (output_.size() == 0) return;
    ostream << std::fixed;
    ostream << "------------------------------------------------------------------------------\n";
    ostream << "Pooling layer size: " << output_.size() << " x " << output_.size() << "\n\n";
    ostream << "Feature map:\n";
    PrintMatrix(output_, ostream, num_decimals);
    ostream << "------------------------------------------------------------------------------\n\n";
}

} /* namespace machine_learning */
} /* namespace yrgo */