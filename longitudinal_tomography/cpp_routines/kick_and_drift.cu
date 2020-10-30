//
// Created by anton on 10/22/20.
//

#include "kick_and_drift.cuh"
#include <iostream>

#define THREADS_PER_BLOCK 512

//#include "sin.h"

#define cudaErrorCheck(exit_code) { gpuAssert((exit_code), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

extern "C" void kick_and_drift(
        double *__restrict__ xp,             // inn/out
        double *__restrict__ yp,             // inn/out
        double *__restrict__ denergy,         // inn
        double *__restrict__ dphi,            // inn
        const double *__restrict__ rf1v,      // inn
        const double *__restrict__ rf2v,      // inn
        const double *__restrict__ phi0,      // inn
        const double *__restrict__ deltaE0,   // inn
        const double *__restrict__ omega_rev0,// inn
        const double *__restrict__ drift_coef,// inn
        const double phi12,
        const double hratio,
        const int dturns,
        const int rec_prof,
        const int nturns,
        const int nparts,
        const int nprofs,
        const bool ftn_out) {

    double *d_xp, *d_yp;
    double *d_denergy, *d_dphi;
    double *d_rf1v, *d_rf2v, *d_phi0, *d_deltaE0, *d_omega_rev0, *d_drift_coef;

    int size_xyp = nparts * nprofs * sizeof(double);
    int size_nparts = nparts * sizeof(double);
    int size_nturns = nturns * sizeof(double);

    cudaErrorCheck( cudaMalloc((void **) &d_xp, size_xyp) );
    cudaErrorCheck( cudaMalloc((void **) &d_yp, size_xyp) );

    cudaErrorCheck( cudaMalloc((void **) &d_denergy, size_nparts) );
    cudaErrorCheck( cudaMalloc((void **) &d_dphi, size_nparts) );

    cudaErrorCheck( cudaMalloc((void **) &d_rf1v, size_nturns) );
    cudaErrorCheck( cudaMalloc((void **) &d_rf2v, size_nturns) );
    cudaErrorCheck( cudaMalloc((void **) &d_phi0, size_nturns) );
    cudaErrorCheck( cudaMalloc((void **) &d_deltaE0, size_nturns) );
    cudaErrorCheck( cudaMalloc((void **) &d_omega_rev0, size_nturns) );
    cudaErrorCheck( cudaMalloc((void **) &d_drift_coef, size_nturns) );

//    cudaMemcpy(d_xp, xp, size_xyp, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_yp, yp, size_xyp, cudaMemcpyHostToDevice);

    cudaErrorCheck( cudaMemcpy(d_denergy, denergy, size_nparts, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_dphi, dphi, size_nparts, cudaMemcpyHostToDevice) );

    cudaErrorCheck( cudaMemcpy(d_rf1v, rf1v, size_nturns, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_rf2v, rf2v, size_nturns, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_phi0, phi0, size_nturns, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_deltaE0, deltaE0, size_nturns, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_omega_rev0, omega_rev0, size_nturns, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_drift_coef, drift_coef, size_nturns, cudaMemcpyHostToDevice) );

    k_d<<<nparts * nprofs/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_xp, d_yp, d_denergy, d_dphi, d_rf1v, d_rf2v,
            d_phi0, d_deltaE0, d_omega_rev0, d_drift_coef,
            phi12, hratio, dturns, rec_prof, nturns, nparts, nprofs, ftn_out);
    cudaErrorCheck( cudaPeekAtLastError() );

//    cudaErrorCheck( cudaDeviceSynchronize() );

    cudaErrorCheck( cudaMemcpy(xp, d_xp, size_xyp, cudaMemcpyDeviceToHost) );
    cudaErrorCheck( cudaMemcpy(yp, d_yp, size_xyp, cudaMemcpyDeviceToHost) );

    cudaErrorCheck( cudaFree(d_xp) );
    cudaErrorCheck( cudaFree(d_yp) );
    cudaErrorCheck( cudaFree(d_denergy) );
    cudaErrorCheck( cudaFree(d_dphi) );
    cudaErrorCheck( cudaFree(d_rf1v) );
    cudaErrorCheck( cudaFree(d_rf2v) );
    cudaErrorCheck( cudaFree(d_phi0) );
    cudaErrorCheck( cudaFree(d_deltaE0) );
    cudaErrorCheck( cudaFree(d_omega_rev0) );
    cudaErrorCheck( cudaFree(d_drift_coef) );
}


__global__ void k_d(double *__restrict__ xp,             // inn/out
                                    double *__restrict__ yp,             // inn/out
                                    double *__restrict__ denergy,         // inn
                                    double *__restrict__ dphi,            // inn
                                    const double *__restrict__ rf1v,      // inn
                                    const double *__restrict__ rf2v,      // inn
                                    const double *__restrict__ phi0,      // inn
                                    const double *__restrict__ deltaE0,   // inn
                                    const double *__restrict__ omega_rev0,// inn
                                    const double *__restrict__ drift_coef,// inn
                                    const double phi12,
                                    const double hratio,
                                    const int dturns,
                                    const int rec_prof,
                                    const int nturns,
                                    const int nparts,
                                    const int nprofs,
                                    const bool ftn_out) {

    int profile = rec_prof;
    int turn = rec_prof * dturns;

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= nparts) {
        return;
    }

    xp[profile * nparts + index] = dphi[index];
    yp[profile * nparts + index] = denergy[index];

    // Upwards
    while (turn < nturns) {
        drift_up(dphi, denergy, drift_coef[turn], nparts, index);

        turn++;

        kick_up(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn], phi12,
                     hratio, nparts, deltaE0[turn], index);

        if (turn % dturns == 0) {
            profile++;

            xp[profile * nparts + index] = dphi[index];
            yp[profile * nparts + index] = denergy[index];
//            if (ftn_out)
//                std::cout << " Tracking from time slice  "
//                          << rec_prof + 1 << " to  " << profile + 1
//                          << ",   0.000% went outside the image width."
//                          << std::endl;
        } //if
    } //while

    profile = rec_prof;
    turn = rec_prof * dturns;

    if (profile > 0) {

        // Going back to initial coordinates
        for (int i = 0; i < nparts; i++) {
            dphi[index] = xp[rec_prof * nparts + index];
            denergy[index] = yp[rec_prof * nparts + index];
        }

        // Downwards
        while (turn > 0) {
            kick_down(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn],
                           phi12, hratio, nparts, deltaE0[turn], index);
            turn--;

            drift_down(dphi, denergy, drift_coef[turn], nparts, index);

            if (turn % dturns == 0) {
                profile--;

                for (int i = 0; i < nparts; i++) {
                    xp[profile * nparts + index] = dphi[index];
                    yp[profile * nparts + index] = denergy[index];
                }
//                if (ftn_out)
//                    std::cout << " Tracking from time slice  "
//                              << rec_prof + 1 << " to  " << profile + 1
//                              << ",   0.000% went outside the image width."
//                              << std::endl;
            }

        }//while
    }
}

__device__ void kick_up(const double *__restrict__ dphi,
                             double *__restrict__ denergy,
                             const double rfv1,
                             const double rfv2,
                             const double phi0,
                             const double phi12,
                             const double hratio,
                             const int nparts,
                             const double acc_kick,
                             const int index) {

    denergy[index] += rfv1 * sin(dphi[index] + phi0)
                      + rfv2 * sin(hratio * (dphi[index] + phi0 - phi12)) - acc_kick;
}

__device__ void kick_down(const double *__restrict__ dphi,
                               double *__restrict__ denergy,
                               const double rfv1,
                               const double rfv2,
                               const double phi0,
                               const double phi12,
                               const double hratio,
                               const int nparts,
                               const double acc_kick,
                               const int index) {

    denergy[index] -= rfv1 * sin(dphi[index] + phi0)
                      + rfv2 * sin(hratio * (dphi[index] + phi0 - phi12)) - acc_kick;
}

__device__ void drift_up(double *__restrict__ dphi,
                              const double *__restrict__ denergy,
                              const double drift_coef,
                              const int nparts,
                              const int index) {

    dphi[index] -= drift_coef * denergy[index];
}

__device__ void drift_down(double *__restrict__ dphi,
                                const double *__restrict__ denergy,
                                const double drift_coef,
                                const int nparts,
                                const int index) {

    dphi[index] += drift_coef * denergy[index];
}

__device__ void calc_xp_and_yp(double *__restrict__ xp,           // inn/out
                                    double *__restrict__ yp,           // inn/out
                                    const double *__restrict__ denergy, // inn
                                    const double *__restrict__ dphi,    // inn
                                    const double phi0,
                                    const double hnum,
                                    const double omega_rev0,
                                    const double dtbin,
                                    const double xorigin,
                                    const double dEbin,
                                    const double yat0,
                                    const int profile,
                                    const int nparts);




//// Calculates the difference in energy between two machine turns.
//// Uses BLonD fast_sin function.
//// Can be called directly from python.
////  Used in hybrid python/C++ class.
//extern "C" void kick_up(const double * __restrict__ dphi,
//                        double * __restrict__ denergy,
//                        const double rfv1,
//                        const double rfv2,
//                        const double phi0,
//                        const double phi12,
//                        const double hratio,
//                        const int nr_particles,
//                        const double acc_kick){
//
//#pragma omp parallel for
//    for (int i=0; i < nr_particles; i++)
//        denergy[i] += rfv1 * vdt::fast_sin(dphi[i] + phi0)
//                      + rfv2 * vdt::fast_sin(hratio * (dphi[i] + phi0 - phi12)) - acc_kick;
//}
//
//extern "C" void kick_down(const double * __restrict__ dphi,
//                          double * __restrict__ denergy,
//                          const double rfv1,
//                          const double rfv2,
//                          const double phi0,
//                          const double phi12,
//                          const double hratio,
//                          const int nr_particles,
//                          const double acc_kick){
//
//#pragma omp parallel for
//    for (int i=0; i < nr_particles; i++)
//        denergy[i] -= rfv1 * vdt::fast_sin(dphi[i] + phi0)
//                      + rfv2 * vdt::fast_sin(hratio * (dphi[i] + phi0 - phi12)) - acc_kick;
//}
//
//// "Drift" function.
//// Calculates the difference in phase between two macine turns.
//// Can be called directly from python.
////  Used in hybrid python/C++ class.
//extern "C" void drift_up(double * __restrict__ dphi,
//                         const double * __restrict__ denergy,
//                         const double drift_coef,
//                         const int nr_particles){
//#pragma omp parallel for
//    for (int i = 0; i < nr_particles; i++)
//        dphi[i] -= drift_coef * denergy[i];
//}
//
//extern "C" void drift_down(double * __restrict__ dphi,
//                           const double * __restrict__ denergy,
//                           const double drift_coef,
//                           const int nr_particles){
//
//#pragma omp parallel for
//    for (int i = 0; i < nr_particles; i++)
//        dphi[i] += drift_coef * denergy[i];
//}
//
//
//// Calculates X and Y coordinates for particles based on a given
////  phase and energy.
//// Can be called directly from python.
//extern "C" void calc_xp_and_yp(double ** __restrict__ xp,           // inn/out
//                               double ** __restrict__ yp,           // inn/out
//                               const double * __restrict__ denergy, // inn
//                               const double * __restrict__ dphi,    // inn
//                               const double phi0,
//                               const double hnum,
//                               const double omega_rev0,
//                               const double dtbin,
//                               const double xorigin,
//                               const double dEbin,
//                               const double yat0,
//                               const int profile,
//                               const int nparts){
//#pragma omp parallel for
//    for(int i=0; i < nparts; i++){
//        xp[profile][i] = (dphi[i] + phi0) / (hnum * omega_rev0 * dtbin) - xorigin;
//        yp[profile][i] = denergy[i] / dEbin + yat0;
//    }//for
//}
//
//extern "C" void kick_and_drift(
//        double ** __restrict__ xp,             // inn/out
//        double ** __restrict__ yp,             // inn/out
//        double * __restrict__ denergy,         // inn
//        double * __restrict__ dphi,            // inn
//        const double * __restrict__ rf1v,      // inn
//        const double * __restrict__ rf2v,      // inn
//        const double * __restrict__ phi0,      // inn
//        const double * __restrict__ deltaE0,   // inn
//        const double * __restrict__ omega_rev0,// inn
//        const double * __restrict__ drift_coef,// inn
//        const double phi12,
//        const double hratio,
//        const int dturns,
//        const int rec_prof,
//        const int nturns,
//        const int nparts,
//        const bool ftn_out){
//    int profile = rec_prof;
//    int turn = rec_prof * dturns;
//
//
//
//
//
//#pragma omp parallel for
//    for(int i=0; i < nparts; i++){
//        xp[profile][i] = dphi[i];
//        yp[profile][i] = denergy[i];
//    }
//
//    // Upwards
//    while(turn < nturns){
//        drift_up(dphi, denergy, drift_coef[turn], nparts);
//
//        turn++;
//
//        kick_up(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn], phi12,
//                hratio, nparts, deltaE0[turn]);
//
//        if (turn % dturns == 0){
//            profile++;
//#pragma omp parallel for
//            for(int i=0; i < nparts; i++){
//                xp[profile][i] = dphi[i];
//                yp[profile][i] = denergy[i];
//            }
//            if (ftn_out)
//                std::cout << " Tracking from time slice  "
//                          << rec_prof + 1 << " to  " << profile + 1
//                          << ",   0.000% went outside the image width."
//                          << std::endl;
//        } //if
//    } //while
//
//    profile = rec_prof;
//    turn = rec_prof * dturns;
//
//    if (profile > 0){
//
//        // Going back to initial coordinates
//#pragma omp parallel for
//        for(int i=0; i < nparts; i++){
//            dphi[i] = xp[rec_prof][i];
//            denergy[i] = yp[rec_prof][i];
//        }
//
//        // Downwards
//        while(turn > 0){
//            kick_down(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn],
//                      phi12, hratio, nparts, deltaE0[turn]);
//            turn--;
//
//            drift_down(dphi, denergy, drift_coef[turn], nparts);
//
//            if (turn % dturns == 0){
//                profile--;
//
//#pragma omp parallel for
//                for(int i=0; i < nparts; i++){
//                    xp[profile][i] = dphi[i];
//                    yp[profile][i] = denergy[i];
//                }
//                if (ftn_out)
//                    std::cout << " Tracking from time slice  "
//                              << rec_prof + 1 << " to  " << profile + 1
//                              << ",   0.000% went outside the image width."
//                              << std::endl;
//            }
//
//        }//while
//    }
//
//}//end func
