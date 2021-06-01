/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file wrappers_cpu.cpp
 *
 * Pybind11 wrappers for tomography C++ routines
 */

#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "data_treatment.h"
#include "kick_and_drift.h"
#include "libtomo.cpu.h"
#include "reconstruct.h"
#include "wrappers.cpu.h"

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
using namespace pybind11::literals;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> d_array;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> i_array;


void CPU::wrapper_kick_up(const d_array &input_dphi,
                          const d_array &input_denergy,
                          const double rf1v,
                          const double rf2v,
                          const double phi0,
                          const double phi12,
                          const double hratio,
                          const int nr_particles,
                          const double acc_kick
) {
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();

    auto *const denergy = static_cast<double *>(denergy_buffer.ptr);
    auto *const dphi = static_cast<double *>(dphi_buffer.ptr);

    tomo::kick_up(dphi, denergy, rf1v, rf2v, phi0, phi12, hratio, nr_particles, acc_kick);
}


void CPU::wrapper_kick_down(const d_array &input_dphi,
                            const d_array &input_denergy,
                            const double rf1v,
                            const double rf2v,
                            const double phi0,
                            const double phi12,
                            const double hratio,
                            const int nr_particles,
                            const double acc_kick
) {
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();

    auto *const denergy = static_cast<double *>(denergy_buffer.ptr);
    auto *const dphi = static_cast<double *>(dphi_buffer.ptr);

    tomo::kick_down(dphi, denergy, rf1v, rf2v, phi0, phi12, hratio, nr_particles, acc_kick);
}


d_array CPU::wrapper_kick(const py::object &machine,
                          const d_array &denergy,
                          const d_array &dphi,
                          const d_array &arr_rfv1,
                          const d_array &arr_rfv2,
                          const int n_particles,
                          const int turn,
                          bool up) {
    d_array arr_phi0 = d_array(machine.attr("phi0"));
    const double phi12 = py::float_(machine.attr("phi12"));
    const double h_ratio = py::float_(machine.attr("h_ratio"));
    d_array arr_deltaE0 = d_array(machine.attr("deltaE0"));

    auto phi0 = arr_phi0.mutable_unchecked<1>();
    auto deltaE0 = arr_deltaE0.mutable_unchecked<1>();
    auto rfv1 = arr_rfv1.unchecked<1>();
    auto rfv2 = arr_rfv2.unchecked<1>();

    if (up)
        wrapper_kick_up(dphi, denergy, rfv1(turn), rfv2(turn),
                        phi0(turn), phi12, h_ratio, n_particles,
                        deltaE0(turn));
    else
        wrapper_kick_down(dphi, denergy, rfv1(turn), rfv2(turn),
                          phi0(turn), phi12, h_ratio, n_particles,
                          deltaE0(turn));

    return denergy;
}


void CPU::wrapper_drift_up(const d_array &input_dphi,
                           const d_array &input_denergy,
                           const double drift_coef,
                           const int n_particles
) {
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();

    auto *const denergy = static_cast<double *>(denergy_buffer.ptr);
    auto *const dphi = static_cast<double *>(dphi_buffer.ptr);

    tomo::drift_up(dphi, denergy, drift_coef, n_particles);
}


void CPU::wrapper_drift_down(const d_array &input_dphi,
                             const d_array &input_denergy,
                             const double drift_coef,
                             const int nr_particles
) {
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();

    auto *const denergy = static_cast<double *>(denergy_buffer.ptr);
    auto *const dphi = static_cast<double *>(dphi_buffer.ptr);

    tomo::drift_down(dphi, denergy, drift_coef, nr_particles);
}


d_array CPU::wrapper_drift(
        const d_array &denergy,
        const d_array &dphi,
        const d_array &input_drift_coef,
        const int n_particles,
        const int turn,
        bool up) {
    auto drift_coef = input_drift_coef.unchecked<1>();

    if (up)
        wrapper_drift_up(dphi, denergy, drift_coef(turn),
                         n_particles);
    else
        wrapper_drift_down(dphi, denergy, drift_coef(turn),
                           n_particles);

    return dphi;
}


// wrap C++ function with NumPy array IO
py::tuple CPU::wrapper_kick_and_drift_machine(
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
        const std::optional<const py::object> callback
) {
    d_array input_phi0 = d_array(machine.attr("phi0"));
    d_array input_deltaE0 = d_array(machine.attr("deltaE0"));
    d_array input_drift_coef = d_array(machine.attr("drift_coef"));
    const double phi12 = py::float_(machine.attr("phi12"));
    const double hratio = py::float_(machine.attr("h_ratio"));
    const int dturns = py::int_(machine.attr("dturns"));

    wrapper_kick_and_drift_scalar(input_xp, input_yp, input_denergy, input_dphi, input_rf1v, input_rf2v,
                                  input_phi0, input_deltaE0, input_drift_coef, phi12, hratio, dturns,
                                  rec_prof, nturns, nparts, ftn_out, callback);

    return py::make_tuple(input_xp, input_yp);
}


py::tuple CPU::wrapper_kick_and_drift_scalar(
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
        const std::optional<const py::object> callback
) {
    double * ptr_phi12 = new double[nturns];
    std::fill_n(ptr_phi12, nturns, phi12);

    py::capsule capsule(ptr_phi12, [] (void* p) {delete[] reinterpret_cast<double*>(p);});
    d_array arr_phi12({nturns}, ptr_phi12, capsule);

    wrapper_kick_and_drift_array(input_xp, input_yp, input_denergy, input_dphi, input_rf1v, input_rf2v, input_phi0, input_deltaE0,
                                 input_drift_coef, arr_phi12, hratio, dturns, rec_prof, nturns, nparts, ftn_out, callback);

    return py::make_tuple(input_xp, input_yp);
}

py::tuple CPU::wrapper_kick_and_drift_array(
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
        const std::optional<const py::object> callback
) {
    py::buffer_info xp_buffer = input_xp.request();
    py::buffer_info yp_buffer = input_yp.request();
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();
    py::buffer_info rf1v_buffer = input_rf1v.request();
    py::buffer_info rf2v_buffer = input_rf2v.request();

    py::buffer_info phi0_buffer = input_phi0.request();
    py::buffer_info deltaE0_buffer = input_deltaE0.request();
    py::buffer_info phi12_buffer = input_phi12.request();
    py::buffer_info drift_coef_buffer = input_drift_coef.request();

    auto *xp = static_cast<double *>(xp_buffer.ptr);
    auto *yp = static_cast<double *>(yp_buffer.ptr);

    const int n_profiles = xp_buffer.shape[0];
    auto **const xp_d = new double *[n_profiles];
    for (int i = 0; i < n_profiles; i++)
        xp_d[i] = &xp[i * nparts];

    auto **const yp_d = new double *[n_profiles];
    for (int i = 0; i < n_profiles; i++)
        yp_d[i] = &yp[i * nparts];

    auto cleanup = [xp_d, yp_d]() {
        delete[] xp_d;
        delete[] yp_d;
    };

    auto *const denergy = static_cast<double *>(denergy_buffer.ptr);
    auto *const dphi = static_cast<double *>(dphi_buffer.ptr);
    const double *const rf1v = static_cast<double *>(rf1v_buffer.ptr);
    const double *const rf2v = static_cast<double *>(rf2v_buffer.ptr);
    const double *const phi0 = static_cast<double *>(phi0_buffer.ptr);
    const double *const deltaE0 = static_cast<double *>(deltaE0_buffer.ptr);
    const double *const phi12 = static_cast<double *>(phi12_buffer.ptr);
    const double *const drift_coef = static_cast<double *>(drift_coef_buffer.ptr);

    std::function<void(int, int)> cb;
    if (callback.has_value()) {
        cb = [&callback](const int progress, const int total) {
            callback.value()(progress, total);
        };
    }
    else
        cb = [] (const int progress, const int total) {(void)progress, (void)total;};

    try {
        kick_and_drift(xp_d, yp_d, denergy, dphi, rf1v, rf2v, phi0, deltaE0, drift_coef,
                       phi12, hratio, dturns, rec_prof, nturns, nparts, ftn_out, cb);
    } catch (const std::exception &e) {
        cleanup();
        throw;
    }

    cleanup();

    return py::make_tuple(input_xp, input_yp);
}

d_array CPU::wrapper_back_project(
        const d_array &input_weights,
        const i_array &input_flat_points,
        const d_array &input_flat_profiles,
        const int n_particles,
        const int n_profiles
) {
    py::buffer_info buffer_weights = input_weights.request();
    py::buffer_info buffer_flat_points = input_flat_points.request();
    py::buffer_info buffer_flat_profiles = input_flat_profiles.request();

    auto *weights = static_cast<double *>(buffer_weights.ptr);
    auto *flat_points = static_cast<int *>(buffer_flat_points.ptr);

    auto *const flat_profiles = static_cast<double *>(buffer_flat_profiles.ptr);

    tomo::back_project(weights, flat_points, flat_profiles, n_particles, n_profiles);

    return input_weights;
}


d_array CPU::wrapper_project(
        const d_array &input_flat_rec,
        const i_array &input_flat_points,
        const d_array &input_weights,
        const int n_particles,
        const int n_profiles,
        const int n_bins
) {
    py::buffer_info buffer_flat_rec = input_flat_rec.request();
    py::buffer_info buffer_flat_points = input_flat_points.request();
    py::buffer_info buffer_weights = input_weights.request();

    auto *weights = static_cast<double *>(buffer_weights.ptr);
    auto *flat_points = static_cast<int *>(buffer_flat_points.ptr);
    auto *const flat_rec = static_cast<double *>(buffer_flat_rec.ptr);

    tomo::project(flat_rec, flat_points, weights, n_particles, n_profiles);

    buffer_flat_rec.shape = std::vector<ssize_t>{n_profiles, n_bins};

    return input_flat_rec;
}


py::tuple CPU::wrapper_reconstruct(
        const i_array &input_xp,
        const d_array &waterfall,
        const int n_iter,
        const int n_bins,
        const int n_particles,
        const int n_profiles,
        const bool verbose,
        const std::optional<const py::object> callback
) {
    py::buffer_info buffer_xp = input_xp.request();
    py::buffer_info buffer_waterfall = waterfall.request();

    auto *weights = new double[n_particles]();
    auto *discr = new double[n_iter + 1]();
    auto *flat_profs = static_cast<double *>(buffer_waterfall.ptr);
    auto *recreated = new double[n_profiles * n_bins]();

    const int *const xp = static_cast<int *>(buffer_xp.ptr);

    std::function<void(int, int)> cb;
    if (callback.has_value()) {
        cb = [&callback](const int progress, const int total) {
            callback.value()(progress, total);
        };
    }
    else
        cb = [] (const int progress, const int total) {(void)progress, (void)total;};

    try {
        reconstruct(weights, xp, flat_profs, recreated, discr, n_iter, n_bins, n_particles, n_profiles, verbose, cb);
    } catch (const std::exception &e) {
        delete[] weights;
        delete[] discr;
        delete[] recreated;

        throw;
    }

    py::capsule capsule_weights(weights, [] (void *p) {delete[] reinterpret_cast<double*>(p);});
    py::capsule capsule_discr(discr, [] (void *p) {delete[] reinterpret_cast<double*>(p);});
    py::capsule capsule_recreated(recreated, [] (void *p) {delete[] reinterpret_cast<double*>(p);});

    py::array_t<double> arr_weights = py::array_t<double>({n_particles}, weights, capsule_weights);
    py::array_t<double> arr_discr = py::array_t<double>({n_iter + 1}, discr, capsule_discr);
    py::array_t<double> arr_recreated = py::array_t<double>({n_profiles, n_bins}, recreated, capsule_recreated);

    return py::make_tuple(arr_weights, arr_discr, arr_recreated);
}


py::array_t<double> CPU::wrapper_make_phase_space(
        const i_array &input_xp,
        const i_array &input_yp,
        const d_array &input_weight,
        const int n_bins
) {
    py::buffer_info buffer_xp = input_xp.request();
    py::buffer_info buffer_yp = input_yp.request();
    py::buffer_info buffer_weight = input_weight.request();

    const int n_particles = buffer_xp.shape[0];

    auto *const xp = static_cast<int *>(buffer_xp.ptr);
    auto *const yp = static_cast<int *>(buffer_yp.ptr);
    auto *const weights = static_cast<double *>(buffer_weight.ptr);

    double *phase_space = make_phase_space(xp, yp, weights, n_particles, n_bins);
    py::capsule capsule(phase_space, [] (void* p) {delete[] reinterpret_cast<double*>(p);});

    return py::array_t<double>({n_bins, n_bins}, phase_space, capsule);
}
