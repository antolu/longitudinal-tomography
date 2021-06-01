/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file libtomo.cpp
 *
 * Pybind11 wrappers for tomography C++ routines
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <algorithm>

#include "docs.h"
#include "libtomo.cpu.h"
#include "wrappers.cpu.h"
#include "reconstruct.h"
#include "data_treatment.h"

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
using namespace pybind11::literals;

// wrap as Python module
PYBIND11_MODULE(libtomo, m) {
    m.doc() = "pybind11 tomo plugin";

    m.def("kick", &CPU::wrapper_kick, kick_docs,
          "machine"_a, "denergy"_a, "dphi"_a,
          "rfv1"_a, "rfv2"_a, "npart"_a, "turn"_a, "up"_a = true);

    m.def("drift", &CPU::wrapper_drift, drift_docs,
          "denergy"_a, "dphi"_a, "drift_coef"_a, "npart"_a, "turn"_a,
          "up"_a = true);

    m.def("kick_up", &CPU::wrapper_kick_up, "Tomography kick up",
          "dphi"_a, "denergy"_a, "rfv1"_a, "rfv2"_a,
          "phi0"_a, "phi12"_a, "h_ratio"_a, "n_particles"_a, "acc_kick"_a);

    m.def("kick_down", &CPU::wrapper_kick_down, "Tomography kick down",
          "dphi"_a, "denergy"_a, "rfv1"_a, "rfv2"_a,
          "phi0"_a, "phi12"_a, "h_ratio"_a, "n_particles"_a, "acc_kick"_a);

    m.def("drift_up", &CPU::wrapper_drift_up, "Tomography drift up",
          "dphi"_a, "denergy"_a, "drift_coef"_a, "n_particles"_a);

    m.def("drift_down", &CPU::wrapper_drift_down, "Tomography drift down",
          "dphi"_a, "denergy"_a, "drift_coef"_a, "n_particles"_a);

    m.def("kick_and_drift", &CPU::wrapper_kick_and_drift_machine, kick_and_drift_docs,
          "xp"_a, "yp"_a, "denergy"_a, "dphi"_a, "rfv1"_a, "rfv2"_a, "machine"_a,
          "rec_prof"_a, "nturns"_a, "nparts"_a, "ftn_out"_a = false, "callback"_a = py::none());

    m.def("kick_and_drift",
          py::overload_cast<const d_array&, const d_array&, const d_array&, const d_array&,
                            const d_array&, const d_array&, const d_array&, const d_array&,
                            const d_array&, const double, const double, const int,
                            const int, const int, const int, const bool,
                            const std::optional<const py::object>
                            >(&CPU::wrapper_kick_and_drift_scalar), kick_and_drift_docs,
          "xp"_a, "yp"_a, "denergy"_a, "dphi"_a, "rfv1"_a, "rfv2"_a, "phi0"_a,
          "deltaE0"_a, "drift_coef"_a, "phi12"_a, "h_ratio"_a, "dturns"_a,
          "rec_prof"_a, "nturns"_a, "nparts"_a, "ftn_out"_a = false, "callback"_a = py::none());

    m.def("kick_and_drift",
          py::overload_cast<const d_array&, const d_array&, const d_array&, const d_array&,
                            const d_array&, const d_array&, const d_array&, const d_array&,
                            const d_array&, const d_array&, const double, const int,
                            const int, const int, const int, const bool,
                            const std::optional<const py::object>
                            >(CPU::wrapper_kick_and_drift_array), kick_and_drift_docs,
          "xp"_a, "yp"_a, "denergy"_a, "dphi"_a, "rfv1"_a, "rfv2"_a, "phi0"_a,
          "deltaE0"_a, "drift_coef"_a, "phi12"_a, "h_ratio"_a, "dturns"_a,
          "rec_prof"_a, "nturns"_a, "nparts"_a, "ftn_out"_a = false, "callback"_a = py::none());

    m.def("project", &CPU::wrapper_project, project_docs,
          "flat_rec"_a, "flat_points"_a, "weights"_a,
          "n_particles"_a, "n_profiles"_a, "n_bins"_a);

    m.def("back_project", &CPU::wrapper_back_project, back_project_docs,
          "weights"_a, "flat_points"_a, "flat_profiles"_a,
          "n_particles"_a, "n_profiles"_a);

    m.def("reconstruct", &CPU::wrapper_reconstruct, reconstruct_docs,
          "xp"_a, "waterfall"_a, "n_iter"_a, "n_bins"_a, "n_particles"_a,
          "n_profiles"_a, "verbose"_a = false, "callback"_a = py::none());

    m.def("make_phase_space", &CPU::wrapper_make_phase_space, make_phase_space_docs,
          "xp"_a, "yp"_a, "weights"_a, "n_bins"_a);
}
