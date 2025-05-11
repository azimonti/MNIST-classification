/************************/
/*     mnist_ga.cpp     */
/*    Version 1.0       */
/*     2025/05/11       */
/************************/

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include "mnist_ga.h"

namespace mnist_ga_trainer
{

    template <typename T>
    void MNIST_GA_Manager<T>::train(nn::ANN_MLP_GA<T>& ga_instance, const std::vector<std::vector<T>>& data,
                                    const std::vector<std::vector<T>>& reference, size_t nGenerations, size_t BatchSize,
                                    bool shuffleTrainingData)
    {
        std::random_device rd;
        std::vector<size_t> s_(data.size());
        std::vector<size_t> f_(ga_instance.GetPopSize());
        std::vector<size_t> v_(ga_instance.GetPopSize());
        typename std::vector<size_t>::iterator it_;
        std::mt19937 g_{rd()};

        std::vector<size_t> network_dimensions = ga_instance.GetNetworkSize();
        if (network_dimensions.empty())
        {
            throw std::runtime_error("GA instance network dimensions are empty in train method.");
        }
        std::vector<T> output_buffer(network_dimensions.back());

        for (size_t i = 0; i < nGenerations; ++i)
        {
            std::iota(s_.begin(), s_.end(), 0);
            if (shuffleTrainingData && !(i % 10))
            {
                std::cout << "Reshuffling training data\n";
                std::shuffle(s_.begin(), s_.end(), g_);
            }

            ga_instance.CreatePopulation();
            std::fill(f_.begin(), f_.end(), 0);

            size_t currentBatchSize = std::min(s_.size(), BatchSize);

            for (size_t j = 0; j < currentBatchSize; ++j)
            {
                it_                                            = s_.begin() + static_cast<ptrdiff_t>(j);
                const std::vector<T>& current_data_sample      = data[*it_];
                const std::vector<T>& current_reference_sample = reference[*it_];

                for (size_t k = 0; k < ga_instance.GetPopSize(); ++k) // k is memberid
                {
                    ga_instance.feedforward(current_data_sample.data(), current_data_sample.size(),
                                            output_buffer.data(), output_buffer.size(), k,
                                            false); // false: full output layer

                    auto max_it_output       = std::max_element(output_buffer.begin(), output_buffer.end());
                    ptrdiff_t max_pos_output = std::distance(output_buffer.begin(), max_it_output);

                    if (current_reference_sample[static_cast<size_t>(max_pos_output)] == static_cast<T>(1)) { f_[k]++; }
                }
            }

            std::iota(v_.begin(), v_.end(), 0);
            std::sort(v_.begin(), v_.end(), [&](size_t idx1, size_t idx2) { return f_[idx1] > f_[idx2]; });

            ga_instance.UpdateWeightsAndBiases(v_);
            ga_instance.UpdateEpochs();

            std::cout << "Current correct ratio: "
                      << static_cast<int>(100.0 * static_cast<T>(f_[v_[0]]) / static_cast<T>(currentBatchSize)) << "% ("
                      << f_[v_[0]] << " / " << currentBatchSize << " training data [shuffle "
                      << (shuffleTrainingData ? "ON" : " OFF") << "])\n";
            std::cout << "Generation " << i + 1 << " / " << nGenerations << " completed (Total "
                      << ga_instance.GetEpochs() << ")\n";
        }
    }

    template <typename T>
    int MNIST_GA_Manager<T>::test(nn::ANN_MLP_GA<T>& ga_instance, const std::vector<std::vector<T>>& data,
                                  const std::vector<std::vector<T>>& reference)
    {
        int iCorrect = 0;
        if (data.empty()) return 0;

        if (ga_instance.GetPopSize() == 0)
        {
            std::cerr << "Warning: GA instance population size is 0 during testing. Attempting to create population."
                      << std::endl;
            ga_instance.CreatePopulation(true);
            if (ga_instance.GetPopSize() == 0)
            {
                std::cerr << "Error: Could not create population for testing." << std::endl;
                return 0;
            }
        }

        std::vector<size_t> network_dimensions_test = ga_instance.GetNetworkSize();
        if (network_dimensions_test.empty())
        {
            std::cerr << "Error: GA instance network dimensions are empty in test method. Cannot determine output "
                         "buffer size."
                      << std::endl;
            return 0;
        }
        std::vector<T> output_buffer(network_dimensions_test.back());

        for (size_t i = 0; i < data.size(); ++i)
        {
            const std::vector<T>& current_data_sample      = data[i];
            const std::vector<T>& current_reference_sample = reference[i];

            if (current_data_sample.size() != network_dimensions_test.front())
            {
                throw std::runtime_error("Input data size mismatch with network input layer size during testing.");
            }

            // Member 0 of the population is assumed to be the best after Deserialize/CreatePopulation.
            ga_instance.feedforward(current_data_sample.data(), current_data_sample.size(), output_buffer.data(),
                                    output_buffer.size(), 0, false); // memberid = 0, false = full output

            auto max_it_output       = std::max_element(output_buffer.begin(), output_buffer.end());
            ptrdiff_t max_pos_output = std::distance(output_buffer.begin(), max_it_output);

            if (current_reference_sample.empty())
            {
                throw std::runtime_error("Reference data empty for current test sample.");
            }
            auto max_it_ref       = std::max_element(current_reference_sample.begin(), current_reference_sample.end());
            ptrdiff_t max_pos_ref = std::distance(current_reference_sample.begin(), max_it_ref);

            if (max_pos_output == max_pos_ref) { iCorrect++; }
        }
        return iCorrect;
    }

    template class MNIST_GA_Manager<float>;
    template class MNIST_GA_Manager<double>;

} // namespace mnist_ga_trainer
