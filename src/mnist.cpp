/************************/
/*      mnist.cpp       */
/*    Version 1.0       */
/*     2023/02/05       */
/************************/

#include <cassert>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include "mnist.h"

namespace fs = std::filesystem;

nn::MNIST::MNIST(const std::string dirname, bool isTraining, bool bVerbose)
    : sTrainingImages(dirname + "/train-images-idx3-ubyte"), sTrainingLabels(dirname + "/train-labels-idx1-ubyte"),
      sTestingImages(dirname + "/t10k-images-idx3-ubyte"), sTestingLabels(dirname + "/t10k-labels-idx1-ubyte")
{
    if (!fs::exists(dirname))
        throw std::runtime_error(std::string("Directory: ").append(dirname).append(" not found. Exiting..."));
#ifdef DEBUG
    if (!fs::exists(sTrainingImages))
        throw std::runtime_error(std::string("File: ").append(sTrainingImages).append(" not found. Exiting..."));
    if (!fs::exists(sTrainingLabels))
        throw std::runtime_error(std::string("File: ").append(sTrainingLabels).append(" not found. Exiting..."));
    if (!fs::exists(sTestingImages))
        throw std::runtime_error(std::string("File: ").append(sTestingImages).append(" not found. Exiting..."));
    if (!fs::exists(sTestingLabels))
        throw std::runtime_error(std::string("File: ").append(sTestingLabels).append(" not found. Exiting..."));
#endif
    if (isTraining)
    {
        ifImages.open(sTrainingImages, std::ios::in | std::ios::binary);
        ifLabels.open(sTrainingLabels, std::ios::in | std::ios::binary);
    }
    else
    {
        ifImages.open(sTestingImages, std::ios::in | std::ios::binary);
        ifLabels.open(sTestingLabels, std::ios::in | std::ios::binary);
    }

    // Read the magic and the meta data
    uint32_t magic_;
    uint32_t nLabels_;
    uint32_t rows_;
    uint32_t cols_;

    ifImages.read(reinterpret_cast<char*>(&magic_), 4);
    magic_ = SwapEndian(magic_);
    if (magic_ != 2051)
        throw std::runtime_error(
            std::string("Incorrect images file magic: ").append(std::to_string(magic_)).append(". Exiting..."));

    ifLabels.read(reinterpret_cast<char*>(&magic_), 4);
    magic_ = SwapEndian(magic_);
    if (magic_ != 2049)
        throw std::runtime_error(
            std::string("Incorrect labels file magic: ").append(std::to_string(magic_)).append(". Exiting..."));

    ifImages.read(reinterpret_cast<char*>(&nItems), 4);
    nItems = SwapEndian(nItems);
    ifLabels.read(reinterpret_cast<char*>(&nLabels_), 4);
    nLabels_ = SwapEndian(nLabels_);

    ifImages.read(reinterpret_cast<char*>(&rows_), 4);
    rows_ = SwapEndian(rows_);
    ifImages.read(reinterpret_cast<char*>(&cols_), 4);
    cols_ = SwapEndian(cols_);

    assert((nItems == 60000) || (nItems == 10000));
    assert((nLabels_ == 60000) || (nLabels_ == 10000));
    assert(nItems == nLabels_);
    assert(rows_ == DIM1);
    assert(cols_ == DIM2);

    if (bVerbose)
    {
        std::cout << "Image and label num is: " << nItems << std::endl;
        std::cout << "Image rows: " << DIM1 << ", cols: " << DIM2 << std::endl;
    }

    // read the images
    vLabelsRaw.resize(nItems);
    for (size_t i = 0; i < nItems; ++i)
    {
        vImagesRaw.emplace_back(std::vector<char>(DIMS));
        vImages.emplace_back(std::vector<size_t>(DIMS));
        vLabels.emplace_back(std::vector<size_t>(10, 0));
    }

    for (size_t i = 0; i < nItems; ++i)
    {
        // read image pixel
        ifImages.read(&vImagesRaw[i][0], DIMS);
        for (size_t k = 0; k < DIM2; ++k)
            for (size_t j = 0; j < DIM1; ++j) vImages[i][k * DIM1 + j] = (vImagesRaw[i][k * DIM1 + j] == 0) ? 0 : 1;
        // read label
        ifLabels.read(&vLabelsRaw[i], 1);
        vLabels[i][size_t(vLabelsRaw[i])] = 1;
    }
}

uint32_t nn::MNIST::SwapEndian(uint32_t val)
{
    uint32_t r = val;
    char* buf  = reinterpret_cast<char*>(&r);
    std::swap(buf[0], buf[3]);
    std::swap(buf[1], buf[2]);
    return r;
}

std::string nn::MNIST::ImageRaw(const size_t n)
{
    std::string ret{};
    for (size_t i = 0; i < DIMS; ++i) ret.append(std::string(1, vImagesRaw[n][i]));
    return ret;
}

size_t nn::MNIST::LabelNumeric(const size_t n)
{
    return static_cast<size_t>(vLabelsRaw[n]);
}

std::string nn::MNIST::Label(const size_t n)
{
    std::string ret;
    ret = vLabelsRaw[n];
    return ret;
}

void nn::MNIST::PrintImage(size_t n)
{
    for (size_t j = 0; j < DIM2; ++j)
    {
        for (size_t i = 0; i < DIM1; ++i) { std::cout << vImages[n][j * DIM1 + i]; }
        std::cout << std::endl;
    }
}

void nn::MNIST::PrintLabel(size_t n)
{
    const auto& vec   = vLabels[n];
    const auto printN = std::min((size_t)10, vec.size());
    for (size_t i = 0; i < printN; ++i) { std::cout << vec[i]; }
    std::cout << std::endl;
}
