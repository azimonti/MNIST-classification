/************************/
/*      main1.cpp       */
/*    Version 1.0       */
/*     2023/02/05       */
/************************/

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include "log/log.h"
#include "mnist.h"

constexpr const char* const IMAGESDIR = "./data/MNIST";

int main()
{
    bool bFileLog = false;
    LOGGER_PARAM(logging::LEVELMAX, logging::INFO);
    LOGGER_PARAM(logging::LOGLINE, true);
    LOGGER_PARAM(logging::LOGTIME, true);
    if (bFileLog)
    {
        LOGGER_PARAM(logging::FILENAME, "out1.log");
        LOGGER_PARAM(logging::FILEOUT, true);
    }
    nn::MNIST img = nn::MNIST(IMAGESDIR, false, true);
    size_t lb     = 36;
    std::cout << "Sample : " << lb << " - Results is: " << img.LabelNumeric(lb) << std::endl;
    img.PrintImage(lb);
    img.PrintLabel(lb);

    return 0;
}
