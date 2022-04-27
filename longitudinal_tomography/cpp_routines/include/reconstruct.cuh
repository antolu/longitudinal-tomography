/**
 * @file reconstruct.cuh
 *
 * @author Anton Lu
 * Contact: anton.lu@cern.ch
 *
 * Functions in pure C/C++ that handles phase space reconstruction.
 * Meant to be called by a Python/C++ wrapper.
 */

#ifndef LIBGPUTOMO_RECONSTRUCT_CUH
#define LIBGPUTOMO_RECONSTRUCT_CUH

#include <functional>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include "pybind11/pybind11.h"

namespace GPU {
    // Back projection using flattened arrays
    void back_project(double *weights,
                      int **flat_points,
                      const double *flat_profiles,
                      const int npart, const int nprof);

    // Projections using flattened arrays
    void project(double *flat_rec,
                 int **flat_points,
                 const double *weights,
                 const int npart, const int nprof);

    void normalize(double *flat_rec,
                   const int nprof,
                   const int nbins);

    __global__ void clip(double *array,
              const int length,
              const double clip_val);

    void find_difference_profile(double *diff_prof,
                                 const double *flat_rec,
                                 const double *flat_profiles,
                                 const int all_bins);

    double discrepancy(const double *diff_prof,
                       const int nprof,
                       const int nbins);

    void compensate_particle_amount(double *diff_prof,
                                    double **rparts,
                                    const int nprof,
                                    const int nbins);

    double max_2d(double **arr,
                  const int x_axis,
                  const int y_axis);

    void count_particles_in_bin(double **rparts,
                                const int **xp,
                                const int nprof,
                                const int npart);

    void reciprocal_particles(double **rparts,
                              const int **xp,
                              const int nbins,
                              const int nprof,
                              const int npart);

    void create_flat_points(const int **xp,
                            int **flat_points,
                            const int npart,
                            const int nprof,
                            const int nbins);

    void reconstruct(double *weights,
                     const int **xp,
                     const double *flat_profiles,
                     double *flat_rec,
                     double *discr,
                     const int niter,
                     const int nbins,
                     const int npart,
                     const int nprof,
                     const bool verbose);

    void reconstruct(double *weights,
                     const int *xp,
                     const double *flat_profiles,
                     double *flat_rec,
                     double *discr,
                     const int niter,
                     const int nbins,
                     const int npart,
                     const int nprof,
                     const bool verbose);

    __global__ void _reconstruct(double *weights,
                                 const int *xp,
                                 const double *flat_profiles,
                                 double *flat_rec,
                                 double *discr,
                                 const int niter,
                                 const int nbins,
                                 const int npart,
                                 const int nprof,
                                 const bool verbose);
}

#endif //LIBGPUTOMO_RECONSTRUCT_H
