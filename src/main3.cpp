/************************/
/*      main3.cpp       */
/*    Version 2.0       */
/*     2025/05/11       */
/************************/

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "log/log.h"
#include "ann_mlp_ga_v1.h"
#include "config_loader.h"
#include "mnist.h"
#include "mnist_ga.h"

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
        LOGGER_PARAM(logging::FILENAME, "out3.log");
        LOGGER_PARAM(logging::FILEOUT, true);
    }

    if (doTraining)
        LOGGER(logging::INFO) << std::string("*** Training mode") + (trainingFromStart ? " from start" : " continue");
    else LOGGER(logging::INFO) << std::string("*** Testing mode");

    if (!std::filesystem::exists(CONFIGFILE))
        throw std::runtime_error(std::string("File: ").append(CONFIGFILE).append(" not found. Exiting..."));

    if (!Config::loadConfiguration(CONFIGFILE))
        throw std::runtime_error(std::string("Could not load configuration file: ").append(CONFIGFILE));

    std::string nname       = Config::getString("nn3.nname");
    std::string archiveFile = Config::getString("nn3.data_file");

    if (doTraining)
    {
        std::string current_set = Config::getString("nn3.current_set");
        std::unique_ptr<nn::ANN_MLP_GA<float>> nn1;
        if (trainingFromStart)
        {
            std::vector<int> nnsize_int = Config::getVectorInt("nn3." + current_set + ".size");
            std::vector<size_t> nnsize(nnsize_int.begin(), nnsize_int.end());
            nn1 = std::make_unique<nn::ANN_MLP_GA<float>>(nnsize);
            // nn1 = std::make_unique<nn::ANN_MLP_GA<float>>(std::vector<size_t>{784, 30, 10});
            nn1->SetName(nname);
            nn1->SetPopulationStrategy(nn::PopulationStrategy::MIXED_WITH_RANDOM_INJECTION, 0.3);
        }
        else
        {
            nn1 = std::make_unique<nn::ANN_MLP_GA<float>>();
            nn1->SetName(nname);
            nn1->Deserialize(archiveFile);
        }
        size_t nGenerations = static_cast<size_t>(Config::getInt("nn3." + current_set + ".nGenerations"));
        size_t BatchSize    = static_cast<size_t>(Config::getInt("nn3." + current_set + ".BatchSize"));

        nn::MNIST imgTrain  = nn::MNIST(IMAGESDIR, true, false);
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

        mnist_ga_trainer::MNIST_GA_Manager<float> ga_manager;
        ga_manager.train(*nn1, images, labels, nGenerations, BatchSize, true);
        // nn1->TrainGA(images, labels, 50, 200, true); // Old call
        nn1->Serialize(archiveFile);
        LOGGER(logging::INFO) << std::string("*** Training completed");
    }
    else
    {
        nn::MNIST imgTest = nn::MNIST(IMAGESDIR, false, false);
        nn::ANN_MLP_GA<float> nn2;
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

        mnist_ga_trainer::MNIST_GA_Manager<float> ga_manager;
        int correct    = ga_manager.test(nn2, images, labels);
        const int size = (int)imgTest.Images().size();
        LOGGER(logging::INFO) << (std::string("*** Correct: ") + std::to_string(correct) + std::string(" / ") +
                                  std::to_string(size) + " (" +
                                  std::to_string(100.0 * static_cast<double>(correct) / size) + " %) ***");
    }

    return 0;
}
