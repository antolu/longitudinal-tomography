#ifndef LIBTOMO_WRAPPERS_CPU_H
#define LIBTOMO_WRAPPERS_CPU_H

/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file wrappers.cpu.cpp
 *
 * Pybind11 wrappers for tomography C++ routine, CPU only
 */

#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
using namespace pybind11::literals;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> d_array;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> i_array;


namespace CPU {
    void wrapper_kick_up(const d_array &input_dphi,
                         const d_array &input_denergy,
                         const double rf1v,
                         const double rf2v,
                         const double phi0,
                         const double phi12,
                         const double hratio,
                         const int nr_particles,
                         const double acc_kick);


    void wrapper_kick_down(const d_array &input_dphi,
                           const d_array &input_denergy,
                           const double rf1v,
                           const double rf2v,
                           const double phi0,
                           const double phi12,
                           const double hratio,
                           const int nr_particles,
                           const double acc_kick);


    d_array wrapper_kick(const py::object &machine,
                         const d_array &denergy,
                         const d_array &dphi,
                         const d_array &arr_rfv1,
                         const d_array &arr_rfv2,
                         const int n_particles,
                         const int turn,
                         bool up);


    void wrapper_drift_up(const d_array &input_dphi,
                          const d_array &input_denergy,
                          const double drift_coef,
                          const int n_particles);


    void wrapper_drift_down(const d_array &input_dphi,
                            const d_array &input_denergy,
                            const double drift_coef,
                            const int nr_particles);


    d_array wrapper_drift(
            const d_array &denergy,
            const d_array &dphi,
            const d_array &input_drift_coef,
            const int n_particles,
            const int turn,
            bool up);


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

    d_array wrapper_back_project(
            const d_array &input_weights,
            const i_array &input_flat_points,
            const d_array &input_flat_profiles,
            const int n_particles,
            const int n_profiles);


    d_array wrapper_project(
            const d_array &input_flat_rec,
            const i_array &input_flat_points,
            const d_array &input_weights,
            const int n_particles,
            const int n_profiles,
            const int n_bins);

    py::tuple wrapper_reconstruct(
            const i_array &input_xp,
            const d_array &waterfall,
            const int n_iter,
            const int n_bins,
            const int n_particles,
            const int n_profiles,
            const bool verbose,
            const std::optional<const py::object> callback);


    py::tuple wrapper_reconstruct_old(
            const d_array &weights,
            const i_array &xp,
            const d_array &flat_profiles,
            const d_array &discr,
            const int n_iter,
            const int n_bins,
            const int n_particles,
            const int n_profiles,
            const bool verbose);


    py::array_t<double> wrapper_make_phase_space(
            const i_array &input_xp,
            const i_array &input_yp,
            const d_array &input_weight,
            const int n_bins);
}

#endif