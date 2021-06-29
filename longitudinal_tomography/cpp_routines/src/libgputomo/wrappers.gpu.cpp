/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file wrappers_cpu.cpp
 *
 * Pybind11 wrappers for tomography C++ routines
 */

#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "wrappers.gpu.h"
#include "include/kick_and_drift.cuh"

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
using namespace pybind11::literals;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> d_array;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> i_array;

// wrap C++ function with NumPy array IO
py::tuple GPU::wrapper_kick_and_drift_machine(
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


py::tuple GPU::wrapper_kick_and_drift_scalar(
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
    double *ptr_phi12 = new double[nturns];
    std::fill_n(ptr_phi12, nturns, phi12);

    py::capsule capsule(ptr_phi12, [](void *p) { delete[] reinterpret_cast<double *>(p); });
    d_array arr_phi12({nturns}, ptr_phi12, capsule);

    wrapper_kick_and_drift_array(input_xp, input_yp, input_denergy, input_dphi, input_rf1v, input_rf2v, input_phi0,
                                 input_deltaE0,
                                 input_drift_coef, arr_phi12, hratio, dturns, rec_prof, nturns, nparts, ftn_out,
                                 callback);

    return py::make_tuple(input_xp, input_yp);
}

py::tuple GPU::wrapper_kick_and_drift_array(
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
    } else
        cb = [](const int progress, const int total) { (void) progress, (void) total; };

    GPU::kick_and_drift(xp, yp, denergy, dphi, rf1v, rf2v, phi0, deltaE0, drift_coef,
                        phi12, hratio, dturns, rec_prof, nturns, nparts, n_profiles, ftn_out);

    return py::make_tuple(input_xp, input_yp);
}
