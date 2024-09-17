cmake_minimum_required(VERSION 3.13.4)
project(nnetworks_example)
set (PROJECT_VERSION "1.0"    )
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION})

# location of the ma-libs
set(MA_LIBS_ROOT "${CMAKE_SOURCE_DIR}/externals/ma-libs")
set(MA_LIBS_CPP_ROOT "${MA_LIBS_ROOT}/cpp")


list(APPEND CMAKE_MODULE_PATH "${MA_LIBS_ROOT}/cmake_modules")
include( COMPILERCOMMON     )
include( COMPILERCPP        )
include( FindLIBS           )
set(CMAKE_CXX_STANDARD 17   )
add_definitions( -DLOGGING -DCOUTEXT )

if(MSVC)
    generic_libs_find(hdf5 ON       )
    generic_libs_find(zlib ON       )
    include_directories( ${HDF5_INCLUDE_DIRS}/src       )
    include_directories( ${HDF5_INCLUDE_DIRS2}/hdf5     )
    link_directories( ${ZLIB_LIBRARY_PATH}              )
else()
    generic_libs_find(hdf5 OFF      )
    add_compile_options(-Wall -Wextra -pedantic -Wconversion -Wno-float-conversion)
endif()

include_directories( ${MATPLOT_INCLUDE_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS}    )
include_directories( ${MA_LIBS_CPP_ROOT}                )
include_directories( ${MA_LIBS_CPP_ROOT}/math           )
include_directories( ${MA_LIBS_CPP_ROOT}/libnn/src      )
include_directories( ${MA_LIBS_CPP_ROOT}/utils          )

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
link_directories( ${HDF5_LIBRARY_PATH}                 )
add_executable( ${PROJECT_NAME} ${SRCS} $<TARGET_OBJECTS:objnnetworks>  )
if(UNITYBUILD)
    set_property( TARGET ${PROJECT_NAME} PROPERTY UNITY_BUILD ON        )
endif()
if(MSVC)
    target_link_libraries( ${PROJECT_NAME}
        debug libhdf5_D zlibstaticd
        optimized libhdf5 zlibstatic )
else()
    target_link_libraries( ${PROJECT_NAME}
        debug hdf5
        optimized hdf5 )
endif()

link_directories( ${MA_LIBS_ROOT}/build/${CMAKE_BUILD_TYPE} )

project(network2_bin)
set ( SRCS ./src/main2.cpp )

link_directories( ${LIBS_DIR}                          )
link_directories( ${HDF5_LIBRARY_PATH}                 )
add_executable( ${PROJECT_NAME} ${SRCS} $<TARGET_OBJECTS:objnnetworks>  )
if(UNITYBUILD)
    set_property( TARGET ${PROJECT_NAME} PROPERTY UNITY_BUILD ON        )
endif()
if(MSVC)
    target_link_libraries( ${PROJECT_NAME}
        debug libhdf5_D debug zlibstaticd debug cpp_nnd
        optimized libhdf5 optimized zlibstatic optimized cpp_nn )
      macro(COPYDLL)
        ADD_CUSTOM_COMMAND( TARGET ${PROJECT_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<$<CONFIG:Release>:../../../externals/ma-libs/build/$<CONFIG>/cpp_alg_lapack.dll>
          $<$<CONFIG:Debug>:../../../externals/ma-libs/build/$<CONFIG>/cpp_alg_lapackd.dll>
          ${CMAKE_BINARY_DIR}/../../$<CONFIG>
          )
        ADD_CUSTOM_COMMAND( TARGET ${PROJECT_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<$<CONFIG:Release>:../../../externals/ma-libs/build/$<CONFIG>/cpp_nn.dll>
          $<$<CONFIG:Debug>:../../../externals/ma-libs/build/$<CONFIG>/cpp_nnd.dll>
          ${CMAKE_BINARY_DIR}/../../$<CONFIG>
          )
      endmacro()
      cmake_language(CALL COPYDLL)
else()
    target_link_libraries( ${PROJECT_NAME}
        debug hdf5 debug cpp_nnd
        optimized hdf5 optimized cpp_nn)
endif()

project(network3_bin)
set ( SRCS ./src/main3.cpp )

link_directories( ${LIBS_DIR}                          )
link_directories( ${HDF5_LIBRARY_PATH}                 )
add_executable( ${PROJECT_NAME} ${SRCS} $<TARGET_OBJECTS:objnnetworks>  )
if(UNITYBUILD)
    set_property( TARGET ${PROJECT_NAME} PROPERTY UNITY_BUILD ON        )
endif()
if(MSVC)
    target_link_libraries( ${PROJECT_NAME}
        debug libhdf5_D debug zlibstaticd debug cpp_nnd
        optimized libhdf5 optimized zlibstatic optimized cpp_nn)
      macro(COPYDLL)
        ADD_CUSTOM_COMMAND( TARGET ${PROJECT_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<$<CONFIG:Release>:../../../externals/ma-libs/build/$<CONFIG>/cpp_alg_lapack.dll>
          $<$<CONFIG:Debug>:../../../externals/ma-libs/build/$<CONFIG>/cpp_alg_lapackd.dll>
          ${CMAKE_BINARY_DIR}/../../$<CONFIG>
          )
        ADD_CUSTOM_COMMAND( TARGET ${PROJECT_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<$<CONFIG:Release>:../../../externals/ma-libs/build/$<CONFIG>/cpp_nn.dll>
          $<$<CONFIG:Debug>:../../../externals/ma-libs/build/$<CONFIG>/cpp_nnd.dll>
          ${CMAKE_BINARY_DIR}/../../$<CONFIG>
          )
      endmacro()
      cmake_language(CALL COPYDLL)
else()
    target_link_libraries( ${PROJECT_NAME}
        debug hdf5 debug cpp_nnd
        optimized hdf5 optimized cpp_nn)
endif()
