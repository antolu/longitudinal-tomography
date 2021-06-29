/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file kick_and_drift.cuh
 *
 * Header file for tracking on GPU
 */

#ifndef LIBGPUTOMO_KICK_AND_DRIFT_CUH
#define LIBGPUTOMO_KICK_AND_DRIFT_CUH

#include <cuda_runtime.h>

namespace GPU {
    void kick_and_drift(
            double *xp,             // inn/out
            double *yp,             // inn/out
            double *denergy,         // inn
            double *dphi,            // inn
            const double *rf1v,      // inn
            const double *rf2v,      // inn
            const double *phi0,      // inn
            const double *deltaE0,   // inn
            const double *drift_coef,// inn
            const double *phi12,
            const double hratio,
            const int dturns,
            const int rec_prof,
            const int nturns,
            const int nparts,
            const int nprofs,
            const bool ftn_out);

    __global__ void k_d(double *xp,             // inn/out
                        double *yp,             // inn/out
                        double *denergy,         // inn
                        double *dphi,            // inn
                        const double *rf1v,      // inn
                        const double *rf2v,      // inn
                        const double *phi0,      // inn
                        const double *deltaE0,   // inn
                        const double *drift_coef,// inn
                        const double *phi12,
                        const double hratio,
                        const int dturns,
                        const int rec_prof,
                        const int nturns,
                        const int nparts,
                        const int nprofs,
                        const bool ftn_out);

    __device__ void kick_up(const double *dphi,
                            double *denergy,
                            const double rfv1,
                            const double rfv2,
                            const double phi0,
                            const double phi12,
                            const double hratio,
                            const int nparts,
                            const double acc_kick,
                            const int index);

    __device__ void kick_down(const double *dphi,
                              double *denergy,
                              const double rfv1,
                              const double rfv2,
                              const double phi0,
                              const double phi12,
                              const double hratio,
                              const int nr_particles,
                              const double acc_kick,
                              const int index);

    __device__ void drift_up(double *dphi,
                             const double *denergy,
                             const double drift_coef,
                             const int nr_particles,
                             const int index);

    __device__ void drift_down(double *dphi,
                               const double *denergy,
                               const double drift_coef,
                               const int nr_particles,
                               const int index);

    __device__ void calc_xp_and_yp(double *xp,           // inn/out
                                   double *yp,           // inn/out
                                   const double *denergy, // inn
                                   const double *dphi,    // inn
                                   const double phi0,
                                   const double hnum,
                                   const double omega_rev0,
                                   const double dtbin,
                                   const double xorigin,
                                   const double dEbin,
                                   const double yat0,
                                   const int profile,
                                   const int nparts);
}

#endif //LIBGPUTOMO_KICK_AND_DRIFT_CUH
