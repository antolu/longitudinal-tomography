#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cmath>

void project(double **  rec,                           // inn/out
             const int ** __restrict__ xp,             // inn
             const double *  __restrict__ weights,     // inn
             const int npart,
             const int nprof,
             const int nbins){

    #pragma acc kernels loop present(xp[:npart][:nprof],\
                                     weights[:npart],\
                                     rec[:nprof][:nbins])\
                             device_type(nvidia) vector_length(32)
    for (int i = 0; i < nprof; i++){
        for (int j = 0; j < npart; j++){
            rec[i][xp[j][i]] += weights[j];
        }
    }
}

void back_project(double *  weights,                     // inn/out
                  const int ** __restrict__ xp,          // inn
                  const double **  profiles,             // inn
                  const int nbins,
                  const int npart,
                  const int nprof){
    double sum_particle;

#pragma acc parallel loop pcopyin(xp[:npart][:nprof],\
                                  profiles[:nprof][:nbins])\
                          pcopy(weights[:npart])\
                          private(sum_particle)\
                          device_type(nvidia) vector_length(32)
#pragma omp parallel for
    for (int i = 0; i < npart; i++){
        sum_particle = 0;
        for (int j = 0; j < nprof; j++)
            sum_particle += profiles[j][xp[i][j]];
        weights[i] += sum_particle;
    }
}

void back_project(double *  weights,               // inn/out
                  const int ** __restrict__ xp,    // inn
                  double **  profiles,             // inn
                  const int nbins,
                  const int npart,
                  const int nprof){
    double sum_particle;

#pragma acc parallel loop pcopyin(xp[:npart][:nprof],\
                                  profiles[:nprof][:nbins])\
                          pcopy(weights[:npart])\
                          private(sum_particle)\
                          device_type(nvidia) vector_length(32)
#pragma omp parallel for
    for (int i = 0; i < npart; i++){
        sum_particle = 0;
        for (int j = 0; j < nprof; j++)
            sum_particle += profiles[j][xp[i][j]];
        weights[i] += sum_particle;
    }
}

void suppress_zeros_norm(double ** __restrict__ rec, // inn/out
                         const int nprof,
                         const int nbins){
    bool positive_flag = false;

    #pragma acc parallel loop present(rec[:nprof][:nbins])\
                              reduction(max:positive_flag)\
                              device_type(nvidia) vector_length(32)
    for(int i=0; i < nprof; i++){
        for(int j=0; j < nbins; j++){
            if(rec[i][j] < 0.0)
                rec[i][j] = 0.0; 
            else
                positive_flag = true;
        }
    }

    if(!positive_flag)
        throw std::runtime_error("All of phase space got reduced to zeroes");

    #pragma acc parallel loop present(rec[:nprof][:nbins])\
                              device_type(nvidia) vector_length(32)
    for(int i=0; i < nprof; i++){
        int j;
        double sum_profile = 0;
        for(j = 0; j < nbins; j++)
            sum_profile += rec[i][j];
        for(j = 0; j < nbins; j++)
            rec[i][j] /= sum_profile;
    }
}

void find_difference_profile(double ** __restrict__ diff_prof,      // out
                             double ** __restrict__ rec,            // inn
                             const double ** __restrict__ profiles, // inn
                             const int nprof,
                             const int nbins){
    #pragma acc parallel loop present(diff_prof[:nprof][:nbins],\
                                     rec[:nprof][:nbins],\
                                     profiles[:nprof][:nbins])\
                             device_type(nvidia) vector_length(32)
    for(int i = 0; i < nprof; i++)
        for(int j = 0; j < nbins; j++)
            diff_prof[i][j] = profiles[i][j] - rec[i][j];
}

double discrepancy(double ** __restrict__ diff_prof,  // inn
                   const int nprof,
                   const int nbins){
    double squared_sum = 0;
    #pragma acc parallel loop present(diff_prof[:nprof][:nbins])\
                              reduction(+:squared_sum)
    for(int i=0; i < nprof; i++)
        for(int j = 0; j < nbins; j++)
            squared_sum += std::pow(diff_prof[i][j], 2.0);
    return std::sqrt(squared_sum / (nprof * nbins));
}

void compensate_particle_amount(double ** __restrict__ diff_prof,       // inn/out
                                double ** __restrict__ rparts,          // inn
                                const int nprof,
                                const int nbins){
    int i, j;
    #pragma acc parallel loop present(diff_prof[:nprof][:nbins],\
                                      rparts[:nprof][:nbins])\
                              device_type(nvidia) vector_length(32)
    for(i=0; i < nprof; i++)
        for(j=0; j < nbins; j++){
            diff_prof[i][j] *= rparts[i][j];  
        }
}

double max_2d(double **  __restrict__ arr,  // inn
              const int x_axis,
              const int y_axis){
    double max_bin_val = 0;
    int i, j;
    for(i=0; i < y_axis; i++)
        for(j=0; j < x_axis; j++)
            if(max_bin_val < arr[i][j])
                max_bin_val = arr[i][j];
    return max_bin_val;
}

void count_particles_in_bin(double ** __restrict__ rparts,      // out
                            const int ** __restrict__ xp,       // inn
                            const int nprof,
                            const int npart){
    int index, i, j;
    for(i=0; i < npart; i++)
        for(j=0; j < nprof; j++){
            index = xp[i][j];
            rparts[j][index] += 1;
        }
}

void reciprocal_particles(double **  __restrict__ rparts,   // out
                          const int ** __restrict__ xp,     // inn
                          const int nbins,
                          const int nprof,
                          const int npart){
    int i, j;

    // initiating rparts to 0
    for(i=0; i < nprof; i++)
        for(j=0; j < nbins; j++)
            rparts[i][j] = 0.0;

    count_particles_in_bin(rparts, xp, nprof, npart);

    int max_bin_val = max_2d(rparts, nbins, nprof);

    // Setting 0's to 1's to avoid zero division
    for(i = 0; i < nprof; i++)
        for(j = 0; j < nbins; j++)
            if(rparts[i][j] == 0.0)
                rparts[i][j] = 1.0;

    // Creating reciprocal
    for(i = 0; i < nprof; i++)
        for(j = 0; j < nbins; j++)
                rparts[i][j] = (double) max_bin_val / rparts[i][j];
}

void print_discr(const double * __restrict__ discr, // inn
                 const int len){
    std::cout << "Discrepancies: " << std::endl;
    for(int i=0; i < len; i++){
        std::cout << i << ": " << discr[i] << " ";
        if ((i + 1) % 4 == 0){
            std::cout << std::endl; 
        }
    }
}

// VERSION 1
// Here i will try with non-flat arrays
extern "C" void reconstruct(double * __restrict__ weights,              // out
                            const int ** __restrict__ xp,               // inn
                            const double ** __restrict__ profiles,      // inn
                            const int niter,
                            const int nbins,
                            const int npart,
                            const int nprof){
    int i;

    // Creating arrays...

    double * discr = new double[niter + 1];
    for(i=0; i < niter + 1; i++)
        discr[i] = 0;

    double** diff_prof = new double*[nprof];    
    double** rec = new double*[nprof];
    double** rparts = new double*[nprof];
    for(i = 0; i < nprof; i++){
        diff_prof[i] = new double[nbins];
        rparts[i] = new double[nbins];
        rec[i] = new double[nbins];
    }

    for (int i = 0; i < nprof; i++)
        for (int j = 0; j < nbins; j++)
            rec[i][j] = 0;

    // Finding the reciprocal of the number of particles in a bin in a given profile.
    // Needed for adjustment of difference-profiles, and correct weighting of particles.
    reciprocal_particles(rparts, xp, nbins, nprof, npart);

#pragma acc data copy(weights[:npart])\
                 pcopyin(xp[:npart][:nprof],\
                         profiles[:nprof][:nbins],\
                         rec[:nprof][:nbins],\
                         rparts[:nprof][:nbins])\
                 pcreate(diff_prof[:nprof][:nbins])
    {

    back_project(weights, xp, profiles, nbins, npart, nprof);

    for(int iteration = 0; iteration < niter; iteration++){
         std::cout << "Iteration: " << iteration + 1 << " of " << niter << std::endl;

        project(rec, xp, weights, npart, nprof, nbins);
        
        suppress_zeros_norm(rec, nprof, nbins);

        find_difference_profile(diff_prof, rec, profiles, nprof, nbins);

        discr[iteration] = discrepancy(diff_prof, nprof, nbins);

        compensate_particle_amount(diff_prof, rparts, nprof, nbins);

        back_project(weights, xp, diff_prof, nbins, npart, nprof);
    } // end for      

    // Calculating final discrepancy
    project(rec, xp, weights, npart, nprof, nbins);
    suppress_zeros_norm(rec, nprof, nbins);
    
    find_difference_profile(diff_prof, rec, profiles, nprof, nbins);
    discr[niter] = discrepancy(diff_prof, nprof, nbins);
    } // end acc data

    // print_discr(discr, niter + 1);

    // Cleaning
    for(i = 0; i < nprof; i++) {
        delete[] rparts[i];
        delete[] rec[i];
        delete[] diff_prof[i];
    }
    delete[] rparts, rec, discr, diff_prof;
}
