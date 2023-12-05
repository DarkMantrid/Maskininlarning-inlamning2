#include <conv_layer.hpp>
#include <algorithm> // For std::max

namespace yrgo {
namespace machine_learning {

ConvLayer::ConvLayer(size_t image_size, size_t kernel_size)
    : image_(image_size, std::vector<double>(image_size, 0.0)),
      kernel_(kernel_size, std::vector<double>(kernel_size, 0.0)),
      output_(),
      kernel_bias_(0.1) { // Initializing kernel_bias_ to a default value of 0.1

    // Initialize the kernel with the desired values
    kernel_ = {
        {0.4, 0.6, 0.7},
        {0.5, 0.6, 0.5},
        {0.6, 0.2, 0.4}
    };
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

    // Perform the convolution operation and apply kernel bias
    PerformConvolution();

    // Apply ReLU activation
    ReLUActivation();

    return true; // Successfully processed the input
}


void ConvLayer::PrintMatrix(const std::vector<std::vector<double>>& data,
                            std::ostream& ostream,
                            const int num_decimals,
                            const size_t offset) {
    for (size_t i = offset; i < data.size(); ++i) {
        for (size_t j = offset; j < data[i].size(); ++j) {
            ostream << std::fixed << std::setprecision(num_decimals) << data[i][j] << " ";
        }
        ostream << "\n";
    }
}

void ConvLayer::Print(std::ostream& ostream, const int num_decimals) {
    if (image_.size() == 0) return;
    ostream << std::fixed;
    ostream << "------------------------------------------------------------------------------\n";
    ostream << "Image size: " << image_.size() << " x " << image_[0].size() << "\n";
    ostream << "Kernel size: " << kernel_.size() << " x " << kernel_[0].size() << "\n\n";

    ostream << "Image:\n";

    // Transpose the image before printing
    std::vector<std::vector<double>> transposed_image(image_[0].size(), std::vector<double>(image_.size(), 0.0));
    for (size_t i = 0; i < image_.size(); ++i) {
        for (size_t j = 0; j < image_[0].size(); ++j) {
            transposed_image[j][i] = image_[i][j];
        }
    }

    PrintMatrix(transposed_image, ostream, num_decimals, 0);

    ostream << "\nKernel:\n";
    PrintMatrix(kernel_, ostream, num_decimals);

    ostream << "\nKernel bias: " << kernel_bias_ << "\n"; // Display kernel bias

    ostream << "\nFeature map:\n";
    PrintMatrix(output_, ostream, num_decimals);
    ostream << "------------------------------------------------------------------------------\n\n";
}


void ConvLayer::PerformConvolution() {
    size_t output_size = image_.size();
    size_t kernel_size = kernel_.size();
    size_t padding = kernel_size / 2;

    // Calculate output size considering zero-padding
    size_t padded_size = output_size + 2 * padding;
    output_.resize(output_size, std::vector<double>(output_size, 0.0));

    std::vector<std::vector<double>> padded_image(padded_size, std::vector<double>(padded_size, 0.0));

    // Copy the input image to the center of the padded image
    for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            padded_image[i + padding][j + padding] = image_[i][j];
        }
    }

    for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < kernel_size; ++k) {
                for (size_t l = 0; l < kernel_size; ++l) {
                    sum += padded_image[i + k][j + l] * kernel_[k][l];
                }
            }
            output_[i][j] = std::max(0.0, sum + kernel_bias_); // Apply kernel bias
        }
    }
}


void ConvLayer::ReLUActivation() {
    for (size_t i = 0; i < output_.size(); ++i) {
        for (size_t j = 0; j < output_[0].size(); ++j) {
            // Apply ReLU activation function
            output_[i][j] = std::max(0.0, output_[i][j]);
        }
    }
}


} /* namespace machine_learning */
} /* namespace yrgo */
