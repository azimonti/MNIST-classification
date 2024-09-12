/************************/
/*      main2.cpp       */
/*    Version 1.0       */
/*     2023/02/05       */
/************************/

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "algebra/matrix.h"
#include "hdf5/hdf5_ext.h"
#include "log/log.h"
#include "ann_mlp_sgd_v1.h"
#include "mnist.h"

constexpr const char* const IMAGESDIR  = "./data/MNIST";
constexpr const char* const CONFIGFILE = "./build/config/nn2.hd5";

int main(int argc, char** argv)
{
    bool doTraining        = true;
    bool trainingFromStart = true;

    if (argc > 1)
    {
        std::string arg = argv[1];
        if (arg == "--training_start")
        {
            doTraining        = true;
            trainingFromStart = true;
        }
        else if (arg == "--training_continue")
        {
            doTraining        = true;
            trainingFromStart = false;
        }
        else if (arg == "--testing") { doTraining = false; }
    }

    bool bFileLog = false;
    LOGGER_PARAM(logging::LEVELMAX, logging::INFO);
    LOGGER_PARAM(logging::LOGLINE, true);
    LOGGER_PARAM(logging::LOGTIME, true);
    if (bFileLog)
    {
        LOGGER_PARAM(logging::FILENAME, "out2.log");
        LOGGER_PARAM(logging::FILEOUT, true);
    }

    if (doTraining)
        LOGGER(logging::INFO) << std::string("*** Training mode") + (trainingFromStart ? " from start" : " continue");
    else LOGGER(logging::INFO) << std::string("*** Testing mode");

    if (!std::filesystem::exists(CONFIGFILE))
        throw std::runtime_error(std::string("File: ").append(CONFIGFILE).append(" not found. Exiting..."));
    h5::H5ppReader h5(CONFIGFILE);
    std::string nname, archiveFile;
    h5.read("nn2/nname", nname);
    h5.read("nn2/data_file", archiveFile);

    if (doTraining)
    {
        std::string current_set;
        h5.read("nn2/current_set", current_set);
        std::unique_ptr<nn::ANN_MLP_SGD<float>> nn1;
        if (trainingFromStart)
        {
            std::vector<size_t> nnsize;
            h5.read("nn2/" + current_set + "/size", nnsize);
            nn1 = std::make_unique<nn::ANN_MLP_SGD<float>>(nnsize);
            // nn1 = std::make_unique<nn::ANN_MLP_SGD<float>>(std::vector<size_t>{784, 30, 10});
            // nn1 = std::make_unique<nn::ANN_MLP_SGD<float>>(std::vector<size_t>{784, 64, 16, 10});
            nn1->SetName(nname);
        }
        else
        {
            nn1 = std::make_unique<nn::ANN_MLP_SGD<float>>();
            nn1->SetName(nname);
            nn1->Deserialize(archiveFile);
        }
        size_t nEpochs, miniBatchSize;
        double eta;
        h5.read("nn2/" + current_set + "/nEpochs", nEpochs);
        h5.read("nn2/" + current_set + "/miniBatchSize", miniBatchSize);
        h5.read("nn2/" + current_set + "/eta", eta);

        nn::MNIST imgTrain = nn::MNIST(IMAGESDIR, true, false);
        std::vector<std::vector<float>> images;
        std::vector<std::vector<float>> labels;
        for (const auto& img : imgTrain.Images())
        {
            std::vector<float> fimg;
            fimg.reserve(img.size());
            for (const auto& val : img) { fimg.push_back(static_cast<float>(val)); }
            images.push_back(fimg);
        }
        for (const auto& lbl : imgTrain.Labels())
        {
            std::vector<float> flbl;
            flbl.reserve(lbl.size());
            for (const auto& val : lbl) { flbl.push_back(static_cast<float>(val)); }
            labels.push_back(flbl);
        }
        nn1->TrainSGD(images, labels, nEpochs, miniBatchSize, eta);
        // nn1->TrainSGD(images, labels, 5, 10, 3.0);
        nn1->Serialize(archiveFile);
        LOGGER(logging::INFO) << std::string("*** Training completed");
    }
    else
    {
        nn::MNIST imgTest = nn::MNIST(IMAGESDIR, false, false);
        nn::ANN_MLP_SGD<float> nn2;
        nn2.SetName(nname);
        nn2.Deserialize(archiveFile);
        std::vector<std::vector<float>> images;
        std::vector<std::vector<float>> labels;
        for (const auto& img : imgTest.Images())
        {
            std::vector<float> fimg;
            fimg.reserve(img.size());
            for (const auto& val : img) { fimg.push_back(static_cast<float>(val)); }
            images.push_back(fimg);
        }
        for (const auto& lbl : imgTest.Labels())
        {
            std::vector<float> flbl;
            flbl.reserve(lbl.size());
            for (const auto& val : lbl) { flbl.push_back(static_cast<float>(val)); }
            labels.push_back(flbl);
        }
        int correct    = nn2.TestSGD(images, labels);
        const int size = (int)imgTest.Images().size();
        LOGGER(logging::INFO) << (std::string("*** Correct: ") + std::to_string(correct) + std::string(" / ") +
                                  std::to_string(size) + " (" +
                                  std::to_string(100.0 * static_cast<double>(correct) / size) + " %) ***");
    }

    return 0;
}
