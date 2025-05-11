#ifndef _MNIST_SGD_H_CA672AC9E7974F35A738E2429759650B_
#define _MNIST_SGD_H_CA672AC9E7974F35A738E2429759650B_

/************************/
/*     mnist_sgd.h      */
/*    Version 1.0       */
/*     2025/05/11       */
/************************/

#include <string>
#include <vector>
#include "ann_mlp_sgd_v1.h"

namespace mnist_sgd_trainer
{

    template <typename T> class MNIST_SGD_Manager
    {
      public:
        MNIST_SGD_Manager() = default;

        void train(nn::ANN_MLP_SGD<T>& sgd_network, const std::vector<std::vector<T>>& data,
                   const std::vector<std::vector<T>>& reference, size_t epochs, size_t miniBatchSize,
                   double eta, // Learning rate
                   bool shuffleTrainingData = true);

        int test(nn::ANN_MLP_SGD<T>& sgd_network, const std::vector<std::vector<T>>& data,
                 const std::vector<std::vector<T>>& reference);
    };

} // namespace mnist_sgd_trainer

#endif
