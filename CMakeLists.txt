cmake_minimum_required(VERSION 3.18)

project(tomography LANGUAGES CUDA CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_STANDARD 17)

if (NOT MSVC)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

include(FindOpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

option(OPTIMIZE_FOR_NATIVE "Build with -march=native" OFF)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

exec_program(${Python_EXECUTABLE}
        ARGS "-c \"import pybind11; print(pybind11.get_include())\""
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        RETURN_VALUE PYBIND11_NOT_FOUND
        )

if (PYBIND11_NOT_FOUND)
    message(FATAL_ERROR "PyBind11 headers not found")
endif ()

include_directories(${Python_INCLUDE_DIRS})
include_directories(${PYBIND11_INCLUDE_DIR})

add_library(libtomo SHARED
    longitudinal_tomography/cpp_routines/libtomo.cpu.cpp
    longitudinal_tomography/cpp_routines/kick_and_drift.cpp
    longitudinal_tomography/cpp_routines/reconstruct.cpp
    longitudinal_tomography/cpp_routines/data_treatment.cpp
    longitudinal_tomography/cpp_routines/wrappers.cpu.cpp)

target_link_libraries(libtomo ${Python_LIBRARIES})

set_target_properties(libtomo
        PROPERTIES
        PREFIX ""
        OUTPUT_NAME "libtomo")

install(TARGETS libtomo
        LIBRARY DESTINATION longitudinal_tomography/cpp_routines)

add_library(libgputomo SHARED
    longitudinal_tomography/cpp_routines/kick_and_drift.cu
    longitudinal_tomography/cpp_routines/libtomo.gpu.cpp
    longitudinal_tomography/cpp_routines/kick_and_drift.cpp
    longitudinal_tomography/cpp_routines/reconstruct.cpp
    longitudinal_tomography/cpp_routines/data_treatment.cpp
    longitudinal_tomography/cpp_routines/wrappers.gpu.cpp
    longitudinal_tomography/cpp_routines/wrappers.cpu.cpp
)

target_link_libraries(libgputomo ${Python_LIBRARIES})

set_target_properties(libgputomo
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 61;75;86
    PREFIX ""
    OUTPUT_NAME "libtomo")
