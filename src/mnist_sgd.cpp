
/************************/
/*     mnist_sgd.cpp    */
/*    Version 1.0       */
/*     2025/05/11       */
/************************/

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include "mnist_sgd.h"

namespace mnist_sgd_trainer
{

    template <typename T>
    void MNIST_SGD_Manager<T>::train(nn::ANN_MLP_SGD<T>& sgd_network, const std::vector<std::vector<T>>& data,
                                     const std::vector<std::vector<T>>& reference, size_t epochs, size_t miniBatchSize,
                                     double eta, bool shuffleTrainingData)
    {
        if (data.empty())
        {
            std::cout << "Training data is empty. Skipping training." << std::endl;
            return;
        }
        if (data.size() != reference.size())
        {
            throw std::invalid_argument("Data and reference counts mismatch in training.");
        }

        std::random_device rd;
        std::vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 g{rd()};

        // Temporary storage for sum of gradients over a mini-batch
        std::vector<la::Matrix<T>> sum_nabla_b;
        std::vector<la::Matrix<T>> sum_nabla_w;

        // Initialize sum_nabla_b and sum_nabla_w structures based on network dimensions
        // This needs to be done once, assuming network structure doesn't change.
        // The ANN_MLP_SGD::backpropagate_calculate_gradients will resize its output params if needed,
        // but for summing, we need them initialized.
        std::vector<size_t> net_size = sgd_network.GetNetworkSize(); // Use original getter, returns a copy
        if (net_size.size() < 2) { throw std::runtime_error("Network has less than 2 layers, cannot train."); }
        sum_nabla_b.resize(net_size.size() - 1);
        sum_nabla_w.resize(net_size.size() - 1);
        for (size_t l = 0; l < net_size.size() - 1; ++l)
        {
            sum_nabla_b[l].Resize(net_size[l + 1], 1);           // net_size is a local copy
            sum_nabla_w[l].Resize(net_size[l + 1], net_size[l]); // net_size is a local copy
        }

        for (size_t i = 0; i < epochs; ++i)
        {
            if (shuffleTrainingData) { std::shuffle(indices.begin(), indices.end(), g); }

            for (size_t j = 0; j < data.size(); j += miniBatchSize)
            {
                // Zero out sum of gradients for the new mini-batch
                for (size_t l = 0; l < sum_nabla_b.size(); ++l)
                {
                    sum_nabla_b[l].Zeros();
                    sum_nabla_w[l].Zeros();
                }

                size_t currentMiniBatchEnd = std::min(j + miniBatchSize, data.size());
                size_t actualMiniBatchSize = currentMiniBatchEnd - j;

                for (size_t k = j; k < currentMiniBatchEnd; ++k)
                {
                    size_t sample_idx = indices[k];

                    // 1. Feedforward
                    sgd_network.feedforward_store_activations(data[sample_idx]);

                    // 2. Backpropagate to get gradients for this sample
                    std::vector<la::Matrix<T>> sample_nabla_b; // Will be resized by backprop
                    std::vector<la::Matrix<T>> sample_nabla_w; // Will be resized by backprop
                    sgd_network.backpropagate_calculate_gradients(reference[sample_idx], sample_nabla_b,
                                                                  sample_nabla_w);

                    // 3. Accumulate gradients
                    for (size_t l = 0; l < sample_nabla_b.size(); ++l)
                    {
                        sum_nabla_b[l] += sample_nabla_b[l];
                        sum_nabla_w[l] += sample_nabla_w[l];
                    }
                }

                // 4. Apply accumulated gradients for the mini-batch
                if (actualMiniBatchSize > 0)
                {
                    sgd_network.apply_gradients(sum_nabla_b, sum_nabla_w,
                                                eta / static_cast<double>(actualMiniBatchSize));
                }
            }

            sgd_network.UpdateEpochs(); // Increment epoch count in the network instance
            std::cout << "Training Epoch " << i + 1 << " / " << epochs << " completed (Total "
                      << sgd_network.GetEpochs() << ")\n";
        }
    }

    template <typename T>
    int MNIST_SGD_Manager<T>::test(nn::ANN_MLP_SGD<T>& sgd_network, const std::vector<std::vector<T>>& data,
                                   const std::vector<std::vector<T>>& reference)
    {
        if (data.empty()) return 0;
        if (data.size() != reference.size())
        {
            throw std::invalid_argument("Data and reference counts mismatch in testing.");
        }

        int iCorrect = 0;
        for (size_t i = 0; i < data.size(); ++i)
        {
            size_t predicted_label_idx = sgd_network.predict(data[i]);

            // Assuming reference[i] is one-hot encoded
            auto max_it_ref            = std::max_element(reference[i].begin(), reference[i].end());
            size_t true_label_idx      = static_cast<size_t>(std::distance(reference[i].begin(), max_it_ref));

            if (predicted_label_idx == true_label_idx) { iCorrect++; }
        }
        return iCorrect;
    }

    // Explicit template instantiation
    template class MNIST_SGD_Manager<float>;
    template class MNIST_SGD_Manager<double>;

} // namespace mnist_sgd_trainer
