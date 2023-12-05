// ConvLayer.cpp

#include <conv_layer.hpp>
#include <algorithm> // For std::max

namespace yrgo {
namespace machine_learning {

ConvLayer::ConvLayer(const size_t image_size, const size_t kernel_size) { 
    // Initialize image and kernel matrices with appropriate sizes
    image_.resize(image_size, std::vector<double>(image_size, 0.0));
    kernel_.resize(kernel_size, std::vector<double>(kernel_size, 0.0));

    // Set kernel bias to a default value
    kernel_bias_ = 0.1;

    // Fill the kernel with example values (you can adjust this according to your needs)
    // This is just an example initialization, you'll need actual values for the kernel
    for (size_t i = 0; i < kernel_size; ++i) {
        for (size_t j = 0; j < kernel_size; ++j) {
            kernel_[i][j] = static_cast<double>((i + 1) * (j + 1)) / 10.0;
        }
    }
}

bool ConvLayer::Feedforward(const std::vector<std::vector<double>>& input) {
    // Ensure input dimensions match the internal image size for processing
    if (input.size() != image_.size() || input[0].size() != image_[0].size()) {
        return false; // Dimensions mismatch
    }

    // Update the image matrix with the input data
    for (size_t i = 0; i < image_.size(); ++i) {
        for (size_t j = 0; j < image_[0].size(); ++j) {
            image_[i][j] = input[i][j];
        }
    }

    // Perform convolution operation
    PerformConvolution();

    // Apply ReLU activation
    ReLUActivation();

    return true; // Successfully processed the input
}

void ConvLayer::PrintMatrix(const std::vector<std::vector<double>>& data,
                            std::ostream& ostream,
                            const int num_decimals,
                            const size_t offset) {
    for (size_t i{offset}; i < data.size() - offset; ++i) {
        for (size_t j{offset}; j < data.size() - offset; ++j) {
            ostream << std::setprecision(num_decimals) << data[j][i] << " ";
        }
        ostream << "\n";
    }
}

void ConvLayer::Print(std::ostream& ostream, const int num_decimals) {
    if (image_.size() == 0) return;
    ostream << std::fixed;
    ostream << "------------------------------------------------------------------------------\n";
    ostream << "Image size: " << image_.size() - 2 << " x " << image_.size() - 2 << "\n";
    ostream << "Kernel size: " << kernel_.size() << " x " << kernel_.size() << "\n\n";

    ostream << "Image:\n";
    PrintMatrix(image_, ostream, num_decimals, 1);
    ostream << "\nKernel:\n";
    PrintMatrix(kernel_, ostream, num_decimals);

    ostream << "\nFeature map:\n";
    PrintMatrix(output_, ostream, num_decimals);
    ostream << "------------------------------------------------------------------------------\n\n";
}

void ConvLayer::PerformConvolution() {
    size_t output_size = image_.size() - kernel_.size() + 1;
    output_.resize(output_size, std::vector<double>(output_size, 0.0));

    for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < kernel_.size(); ++k) {
                for (size_t l = 0; l < kernel_.size(); ++l) {
                    sum += image_[i + k][j + l] * kernel_[k][l];
                }
            }
            output_[i][j] = sum + kernel_bias_;
        }
    }
}

void ConvLayer::ReLUActivation() {
    for (size_t i = 0; i < output_.size(); ++i) {
        for (size_t j = 0; j < output_[0].size(); ++j) {
            output_[i][j] = std::max(0.0, output_[i][j]);
        }
    }
}

} /* namespace machine_learning */
} /* namespace yrgo */
