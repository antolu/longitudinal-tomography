/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file data_treatment.h
 *
 * C++ equivalent of the `longitudinal_tomography.data.data_treatment` module.
 */

#ifndef LIBTOMO_DATA_TREATMENT_H
#define LIBTOMO_DATA_TREATMENT_H

double *make_phase_space(const int *const xp, const int *const yp, const double *const weight, const int n_particles,
                         const int n_bins);

#endif