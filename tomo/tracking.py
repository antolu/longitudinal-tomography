import numpy as np
from numba import njit
from cpp_routines.tomolib_wrappers import kick, drift


class Tracking:

    def __init__(self, ts, mi):
        self.mapinfo = mi
        self.timespace = ts

    def track(self):
        nr_of_particles = self.find_nr_of_particles()

        xp = np.zeros((self.timespace.par.profile_count, nr_of_particles))
        yp = np.copy(xp)

        # Creating a homogeneous distribution of particles
        xp[0], yp[0] = self._initiate_points()

        # Calculating radio frequency voltages for each turn
        rf1v = (self.timespace.par.vrf1
                + self.timespace.par.vrf1dot
                * self.timespace.par.time_at_turn) * self.timespace.par.q
        rf2v = (self.timespace.par.vrf2
                + self.timespace.par.vrf2dot
                * self.timespace.par.time_at_turn) * self.timespace.par.q

        # Retrieving some numbers for array creation

        nr_of_turns = (self.timespace.par.dturns
                       * (self.timespace.par.profile_count - 1))

        dphi, denergy = self.calc_dphi_denergy(xp[0], yp[0])

        dphi = np.ascontiguousarray(dphi)
        denergy = np.ascontiguousarray(denergy)
        rf1v = np.ascontiguousarray(rf1v)
        rf2v = np.ascontiguousarray(rf2v)

        xp, yp = self.kick_and_drift(xp, yp, denergy, dphi,
                                     rf1v, rf2v, nr_of_turns, nr_of_particles)

        xp = np.ascontiguousarray(xp)
        yp = np.ascontiguousarray(yp)

        return np.ceil(xp).astype(int), np.ceil(yp).astype(int)

    def kick_and_drift(self, xp, yp, denergy, dphi, rf1v, rf2v,
                       n_turns, n_part):
        turn = 0
        profile = 0
        print(f'tracking to profile {profile + 1}')
        while turn < n_turns:
            # Calculating change in phase for each particle at a turn
            dphi = drift(denergy, dphi, self.timespace.par.dphase,
                         n_part, turn)
            turn += 1
            # Calculating change in energy for each particle at a turn
            denergy = kick(self.timespace.par, denergy, dphi, rf1v, rf2v,
                           n_part, turn)

            if turn % self.timespace.par.dturns == 0:
                profile += 1
                xp[profile] = ((dphi + self.timespace.par.phi0[turn])
                               / (float(self.timespace.par.h_num)
                               * self.timespace.par.omega_rev0[turn]
                               * self.timespace.par.dtbin)
                               - self.timespace.par.x_origin)
                yp[profile] = (denergy / float(self.mapinfo.dEbin)
                               + self.timespace.par.yat0)
                print(f'tracking to profile {profile + 1}')
        return xp, yp

    def find_nr_of_particles(self):
        jdiff = (self.mapinfo.jmax
                 - self.mapinfo.jmin)
        pxls = np.sum(jdiff[self.mapinfo.imin
                            :self.mapinfo.imax + 1])
        return int(pxls * self.timespace.par.snpt**2)

    def calc_dphi_denergy(self, xp, yp, turn=0):
        tpar = self.timespace.par
        dphi = ((xp + tpar.x_origin)
                * tpar.h_num
                * tpar.omega_rev0[turn]
                * tpar.dtbin
                - tpar.phi0[turn])
        denergy = (yp - tpar.yat0) * self.mapinfo.dEbin
        return dphi, denergy

    def _populate_bins(self, sqrtNbrPoints):
        xCoords = ((2.0 * np.arange(1, sqrtNbrPoints + 1) - 1)
                   / (2.0 * sqrtNbrPoints))
        yCoords = xCoords

        xCoords = xCoords.repeat(sqrtNbrPoints, 0).reshape(
            (sqrtNbrPoints, sqrtNbrPoints))
        yCoords = np.repeat([yCoords], sqrtNbrPoints, 0)
        return [xCoords, yCoords]

    # Wrapper function for creating homogeneously distributed particles
    def _initiate_points(self):
        # Initializing points for homogeneous distr. particles
        points = self._populate_bins(self.timespace.par.snpt)

        xp = np.zeros(self.find_nr_of_particles())
        yp = np.copy(xp)

        # Creating the first profile with equally distributed points
        (xp,
         yp) = self._init_tracked_point(
                                self.timespace.par.snpt, self.mapinfo.imin,
                                self.mapinfo.imax, self.mapinfo.jmin,
                                self.mapinfo.jmax, xp,
                                yp, points[0], points[1])

        return xp, yp

    @staticmethod
    @njit
    # Creating homogeneously distributed particles
    def _init_tracked_point(snpt, imin, imax,
                            jmin, jmax, xp, yp,
                            xpoints, ypoints):
        k = 0
        for iLim in range(imin, imax + 1):
            for jLim in range(jmin[iLim], jmax[iLim]):
                for i in range(snpt):
                    for j in range(snpt):
                        xp[k] = iLim + xpoints[i, j]
                        yp[k] = jLim + ypoints[i, j]
                        k += 1
        return xp, yp
