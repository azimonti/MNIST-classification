#ifndef _MNIST_GA_H_872075BC76B743319C566A5BC2FB64B6_
#define _MNIST_GA_H_872075BC76B743319C566A5BC2FB64B6_

/************************/
/*     mnist_ga.h       */
/*    Version 1.0       */
/*     2025/05/11       */
/************************/

#include <string>
#include <vector>
#include "ann_mlp_ga_v1.h"

namespace mnist_ga_trainer
{

    template <typename T> class MNIST_GA_Manager
    {
      public:
        MNIST_GA_Manager() = default;

        void train(nn::ANN_MLP_GA<T>& ga_instance, const std::vector<std::vector<T>>& data,
                   const std::vector<std::vector<T>>& reference, size_t nGenerations, size_t BatchSize,
                   bool shuffleTrainingData = true);

        int test(nn::ANN_MLP_GA<T>& ga_instance, const std::vector<std::vector<T>>& data,
                 const std::vector<std::vector<T>>& reference);
    };

} // namespace mnist_ga_trainer

#endif
