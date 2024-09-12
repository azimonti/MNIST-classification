#ifndef _MNIST_H_122FDE746A84469CAF3080DBA4C76C11_
#define _MNIST_H_122FDE746A84469CAF3080DBA4C76C11_

/************************/
/*     mnist.h          */
/*    Version 1.0       */
/*     2023/02/05       */
/************************/

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace nn
{
    class MNIST
    {
        // images are 28 x 28 pixels
        static constexpr size_t DIM1 = 28;
        static constexpr size_t DIM2 = 28;
        static constexpr size_t DIMS = 784;

      public:
        MNIST(const std::string dirname, bool isTraining = true, bool bVerbose = false);

        std::vector<std::vector<size_t>>& Images() { return vImages; }

        std::vector<std::vector<size_t>>& Labels() { return vLabels; }

        std::string ImageRaw(const size_t n);
        std::string Label(const size_t n);
        size_t LabelNumeric(const size_t n);
        void PrintImage(size_t n);
        void PrintLabel(size_t n);

      private:
        uint32_t SwapEndian(uint32_t val);
        const std::string sTrainingImages;
        const std::string sTrainingLabels;
        const std::string sTestingImages;
        const std::string sTestingLabels;
        uint32_t nItems;
        std::ifstream ifImages;
        std::ifstream ifLabels;
        std::vector<std::vector<char>> vImagesRaw;
        std::vector<std::vector<size_t>> vImages;
        std::vector<char> vLabelsRaw;
        std::vector<std::vector<size_t>> vLabels;
    };

} // namespace nn

#endif
