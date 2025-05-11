cmake_minimum_required(VERSION 3.13.4)
project(nnetworks_example)
set (PROJECT_VERSION "1.0"    )
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION})

# location of ma-libs
set(MA_LIBS_ROOT "${CMAKE_SOURCE_DIR}/externals/ma-libs")
set(MA_LIBS_CPP_ROOT "${MA_LIBS_ROOT}/cpp")


list(APPEND CMAKE_MODULE_PATH "${MA_LIBS_ROOT}/cmake_modules")
include( COMPILERCOMMON     )
include( COMPILERCPP        )
include( FindLIBS           )
set(CMAKE_CXX_STANDARD 17   )
add_definitions( -DLOGGING -DCOUTEXT )

include_directories( "${MA_LIBS_CPP_ROOT}"                            )
include_directories( "${MA_LIBS_CPP_ROOT}/math"                       )
include_directories( "${MA_LIBS_CPP_ROOT}/libnn/src"                  )
include_directories( "${MA_LIBS_CPP_ROOT}/libconfig_loader/src"       )
include_directories( "${MA_LIBS_CPP_ROOT}/utils"                      )

add_definitions( -DLOGGING )

set ( SRCS
    ./src/mnist.cpp
    ${MA_LIBS_CPP_ROOT}/utils/log/log.cpp
    )

add_library( objnnetworks OBJECT ${SRCS}                               )
set_property( TARGET objnnetworks PROPERTY POSITION_INDEPENDENT_CODE 1 )
if(UNITYBUILD)
    set_property( TARGET objnnetworks PROPERTY UNITY_BUILD ON          )
endif()

project(network1_bin)
set ( SRCS ./src/main1.cpp )

link_directories( ${LIBS_DIR}                          )
# link_directories( ${HDF5_LIBRARY_PATH}                 ) # Removed HDF5
add_executable( ${PROJECT_NAME} ${SRCS} $<TARGET_OBJECTS:objnnetworks>  )
if(UNITYBUILD)
    set_property( TARGET ${PROJECT_NAME} PROPERTY UNITY_BUILD ON        )
endif()

link_directories( ${MA_LIBS_ROOT}/build/${CMAKE_BUILD_TYPE} )

project(network2_bin)
set ( SRCS ./src/main2.cpp ./src/mnist_sgd.cpp )

link_directories( ${LIBS_DIR}                          )
# link_directories( ${HDF5_LIBRARY_PATH}                 ) # Removed HDF5
add_executable( ${PROJECT_NAME} ${SRCS} $<TARGET_OBJECTS:objnnetworks>  )
if(UNITYBUILD)
    set_property( TARGET ${PROJECT_NAME} PROPERTY UNITY_BUILD ON        )
endif()

target_link_libraries( ${PROJECT_NAME} debug cpp_nn_blasd debug config_loaderd
  optimized cpp_nn_blas optimized config_loader)

project(network3_bin)
set ( SRCS ./src/main3.cpp ./src/mnist_ga.cpp )

link_directories( ${LIBS_DIR}                          )
# link_directories( ${HDF5_LIBRARY_PATH}                 ) # Removed HDF5
add_executable( ${PROJECT_NAME} ${SRCS} $<TARGET_OBJECTS:objnnetworks>  )
if(UNITYBUILD)
    set_property( TARGET ${PROJECT_NAME} PROPERTY UNITY_BUILD ON        )
endif()

target_link_libraries( ${PROJECT_NAME} debug cpp_nn_blasd debug config_loaderd
  optimized cpp_nn_blas optimized config_loader)
