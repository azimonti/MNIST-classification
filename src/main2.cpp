/************************/
/*      main2.cpp       */
/*    Version 2.0       */
/*     2025/05/11       */
/************************/

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "algebra/matrix.h"
#include "log/log.h"
#include "ann_mlp_sgd_v1.h"
#include "config_loader.h"
#include "mnist.h"
#include "mnist_sgd.h"

constexpr const char* const IMAGESDIR  = "./data/MNIST";
constexpr const char* const CONFIGFILE = "./config.txt";

int main(int argc, char** argv)
{
    bool doTraining        = true;
    bool trainingFromStart = true;

    if (argc > 1)
    {
        std::string arg = argv[1];
        if (arg == "--start_training")
        {
            doTraining        = true;
            trainingFromStart = true;
        }
        else if (arg == "--continue_training")
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

    if (!Config::loadConfiguration(CONFIGFILE))
        throw std::runtime_error(std::string("Could not load configuration file: ").append(CONFIGFILE));

    std::string nname       = Config::getString("nn2.nname");
    std::string archiveFile = Config::getString("nn2.data_file");

    if (doTraining)
    {
        std::string current_set = Config::getString("nn2.current_set");
        std::unique_ptr<nn::ANN_MLP_SGD<float>> nn1;
        if (trainingFromStart)
        {
            std::vector<int> nnsize_int = Config::getVectorInt("nn2." + current_set + ".size");
            std::vector<size_t> nnsize(nnsize_int.begin(), nnsize_int.end());
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
        size_t nEpochs       = static_cast<size_t>(Config::getInt("nn2." + current_set + ".nEpochs"));
        size_t miniBatchSize = static_cast<size_t>(Config::getInt("nn2." + current_set + ".miniBatchSize"));
        double eta           = Config::getDouble("nn2." + current_set + ".eta");

        nn::MNIST imgTrain   = nn::MNIST(IMAGESDIR, true, false);
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

        mnist_sgd_trainer::MNIST_SGD_Manager<float> sgd_manager;
        sgd_manager.train(*nn1, images, labels, nEpochs, miniBatchSize, eta,
                          true); // Added shuffle=true, adjust if needed
        // nn1->TrainSGD(images, labels, 5, 10, 3.0); // Old call
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

        mnist_sgd_trainer::MNIST_SGD_Manager<float> sgd_manager;
        int correct    = sgd_manager.test(nn2, images, labels);
        const int size = (int)imgTest.Images().size();
        LOGGER(logging::INFO) << (std::string("*** Correct: ") + std::to_string(correct) + std::string(" / ") +
                                  std::to_string(size) + " (" +
                                  std::to_string(100.0 * static_cast<double>(correct) / size) + " %) ***");
    }

    return 0;
}
