#ifndef LIBTOMO_WRAPPERS_GPU_H
#define LIBTOMO_WRAPPERS_GPU_H

/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file wrappers_cpu.cpp
 *
 * Pybind11 wrappers for tomography C++ routines
 */

#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "include/kick_and_drift.cuh"

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
using namespace pybind11::literals;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> d_array;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> i_array;

namespace GPU {

    // wrap C++ function with NumPy array IO
    py::tuple wrapper_kick_and_drift_machine(
            const d_array &input_xp,
            const d_array &input_yp,
            const d_array &input_denergy,
            const d_array &input_dphi,
            const d_array &input_rf1v,
            const d_array &input_rf2v,
            const py::object &machine,
            const int rec_prof,
            const int nturns,
            const int nparts,
            const bool ftn_out,
            const std::optional<const py::object> callback);


    py::tuple wrapper_kick_and_drift_scalar(
            const d_array &input_xp,
            const d_array &input_yp,
            const d_array &input_denergy,
            const d_array &input_dphi,
            const d_array &input_rf1v,
            const d_array &input_rf2v,
            const d_array &input_phi0,
            const d_array &input_deltaE0,
            const d_array &input_drift_coef,
            const double phi12,
            const double hratio,
            const int dturns,
            const int rec_prof,
            const int nturns,
            const int nparts,
            const bool ftn_out,
            const std::optional<const py::object> callback);


    py::tuple wrapper_kick_and_drift_array(
            const d_array &input_xp,
            const d_array &input_yp,
            const d_array &input_denergy,
            const d_array &input_dphi,
            const d_array &input_rf1v,
            const d_array &input_rf2v,
            const d_array &input_phi0,
            const d_array &input_deltaE0,
            const d_array &input_drift_coef,
            const d_array &input_phi12,
            const double hratio,
            const int dturns,
            const int rec_prof,
            const int nturns,
            const int nparts,
            const bool ftn_out,
            const std::optional<const py::object> callback);
}

#endif