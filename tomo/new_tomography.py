import matplotlib.pyplot as plt
import numpy as np
import time as tm
import logging as lg
from numba import njit, prange


class NewTomography:

    def __init__(self, timespace, tracked_points):
        self.ts = timespace
        self.tracked_points = tracked_points - 1  # Fortran compensation
        # self.weights = np.zeros((self.ts.par.profile_count,
        #                          tracked_points.shape[0]))
        # self.bins = np.zeros((self.ts.par.profile_length,
        #                       self.ts.par.profile_count))

    def run3(self):
        weights = np.zeros(self.tracked_points.shape[0])

        t0 = tm.perf_counter()
        self.calc_weights(weights, self.ts.profiles, self.tracked_points)
        print(tm.perf_counter() - t0)



    @staticmethod
    @njit
    def calc_weights(weights, profiles, points):
        for i in range(len(weights)):
            for prof, po in enumerate(points[i]):
                weights[i] += profiles[prof, po]

    def run(self):

        diff_prof = np.zeros(self.ts.profiles.shape)

        self.back_project_njit(self.tracked_points,
                               self.ts.profiles,
                               self.weights,
                               self.ts.par.profile_count)

        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')

            t0 = tm.perf_counter()
            self.project_njit(self.tracked_points, self.bins,
                              self.weights,
                              self.ts.par.profile_count)
            lg.info('project_t: ' + str(tm.perf_counter() - t0))

            self.weights[:, :] = 0

            t0 = tm.perf_counter()
            for p in range(self.ts.par.profile_count):
                diff_prof[p] = ((self.ts.profiles[p]
                                 / np.sum(self.ts.profiles[p]))
                                - (self.bins[:, p]
                                / np.sum(self.bins[:, p])))
            lg.info('difference_t: ' + str(tm.perf_counter() - t0))

            lg.info(f'discrepancy: {self.discrepancy(diff_prof)}')

            t0 = tm.perf_counter()
            self.back_project_njit(self.tracked_points,
                                   diff_prof,
                                   self.bins,
                                   self.ts.par.profile_count)
            lg.info('back_projection_t: ' + str(tm.perf_counter() - t0))

        # TEMP
        # self.analyze(profilei=0, diff_prof=diff_prof)
        # END TEMP

    def run2(self):
        diff_prof = np.zeros(self.ts.profiles.shape)

        self.back_project_njit2(self.tracked_points,
                                self.ts.profiles,
                                self.bins,
                                self.ts.par.profile_count)

        print('Iterating...')

        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')

            t0 = tm.perf_counter()
            for p in range(self.ts.par.profile_count):
                diff_prof[p] = ((self.ts.profiles[p]
                                 / np.sum(self.ts.profiles[p]))
                                - (self.bins[:, p]
                                   / np.sum(self.bins[:, p])))
            print(f'difference time: {tm.perf_counter() - t0}')

            # TEMP
            # self.analyze(profilei=0, diff_prof=diff_prof)
            # END TEMP

            t0 = tm.perf_counter()
            self.back_project_njit2(self.tracked_points,
                                    diff_prof,
                                    self.bins,
                                    self.ts.par.profile_count)
            print(f'projection time: {tm.perf_counter() - t0}')

            print(f'discrepancy: {self.discrepancy(diff_prof)}')

        # TEMP
        self.analyze(profilei=0, diff_prof=diff_prof)
        # END TEMP

    @staticmethod
    @njit(parallel=True)
    def project_njit(tracked_points, bins, weights, profile_count):
        for profile in prange(profile_count):
            for point in range(tracked_points.shape[0]):
                bins[tracked_points[point, profile],
                     profile] += weights[profile, point]

    @staticmethod
    @njit(parallel=True)
    def back_project_njit(tracked_points,
                          profiles,
                          weight_factors,
                          profile_count):

        for profile in prange(profile_count):
            counter = 0
            for point in tracked_points[:, profile]:
                weight_factors[profile, counter] += profiles[profile, point]
                counter += 1

    @staticmethod
    @njit(parallel=True)
    def back_project_njit2(tracked_points,
                           profiles,
                           bins,
                           profile_count):
        for profile in prange(profile_count):
            for index, point in enumerate(tracked_points[:, profile]):
                bins[tracked_points[index, profile], profile] += profiles[profile, point]

    def discrepancy(self, diff_profiles):
        return np.sqrt(np.sum(diff_profiles**2)/(self.ts.par.profile_length
                                                 * self.ts.par.profile_count))

    # TEMP
    def analyze(self, profilei, diff_prof):
        plt.figure()

        # Plotting profiles
        plt.subplot(311)
        plt.title('Profiles')
        plt.plot(self.ts.profiles[profilei] / np.sum(self.ts.profiles[profilei]))
        plt.plot(self.bins[:, profilei] / np.sum(self.bins[:, profilei]))
        plt.gca().legend(('original', 'recreated python'))

        # Plotting difference
        plt.subplot(313)
        plt.title('Difference')
        plt.plot(diff_prof[profilei])
        plt.show()
    # END TEMP
