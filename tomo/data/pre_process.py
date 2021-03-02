"""
Module containing functions for pre-processing raw data

The functions here are to be moved over from tomo.data.data_treatment

:Author(s): **Anton Lu**, **Christoffer Hjertø Grindheim**
"""
from typing import Union, Tuple, List

import numpy as np
from multipledispatch import dispatch
from scipy import optimize

from tomo.utils import physics
from .profiles import Profiles
from .. import assertions as asrt
from ..tracking import Machine
from ..tracking.machine_base import MachineABC


def rebin(waterfall: np.ndarray, rbn: int, dtbin: float = None,
          synch_part_x: float = None) \
        -> Union[Tuple[np.ndarray, float, float], Tuple[np.ndarray, float]]:
    """Rebin waterfall from shape (P, X) to (P, Y).
    P is the number of profiles, X is the original number of bins,
    and Y is the number of bins after the re-binning.

    The algorithm is based on the rebinning function from the
    original tomography program.

    An array of length N, rebinned with a rebin factor of R will result
    in a array of length N / R. If N is not dividable on R, the resulting
    array will have the length (N / R) + 1.

    Parameters
    ----------
    waterfall: ndarray
        2D array of raw-data shaped as waterfall. Shape: (nprofiles, nbins).
    rbn: int
        Rebinning factor.
    dtbin: float
        Size of profile bins [s]. If provided, the function
        will return the new size of the bins after rebinning.
    synch_part_x: float
        x-coordinate of synchronous particle, measured in bins.
        If provided, the function will return its updated coordinate.

    Returns
    -------
    rebinned: ndarray
        Rebinned waterfall
    dtbin: float, optional, default=None
        If a dtbin has been provided in the arguments, the
        new size of the profile bins will be returned. Otherwise,
        None will be returned.
    synch_part_x: float, optional, default=None
        If a synch_part_x has been provided in the arguments, the
        new x-coordinate of the synchronous particle in bins will be returned.
        If unfilled only rebinned and dtbin will be returned
    """
    data = np.copy(waterfall)

    # Check that there is enough data to for the given rebin factor.
    if data.shape[1] % rbn == 0:
        rebinned = _rebin_dividable(data, rbn)
    else:
        rebinned = _rebin_individable(data, rbn)

    if dtbin is not None:
        dtbin *= rbn
    if synch_part_x is not None:
        synch_part_x /= rbn
        return rebinned, dtbin, synch_part_x

    return rebinned, dtbin


@dispatch(np.ndarray, MachineABC)
def fit_synch_part_x(waterfall: np.ndarray, machine: MachineABC)\
        -> Tuple[np.ndarray, float, float]:
    """Linear fit to estimate the phase coordinate of the synchronous
    particle. The found phase is returned as a x-coordinate of the phase space
    coordinate systems in fractions of bins. The estimation is done at
    the beam reference profile, which is set in the
    :class:`tomo.tracking.machine.Machine` object.

    Parameters
    ----------
    waterfall : np.ndarray
        2D array of raw-data shaped as waterfall. Shape: (nprofiles, nbins).
    machine : Machine
        Holds all information needed for the particle tracking and the
        generation of initial the particle distribution.

    Returns
    -------
    fitted_synch_part_x
        X coordinate in the phase space coordinate system of the synchronous
        particle given in bin numbers.
    lower bunch limit
        Estimation of the lower bunch limit in bin numbers.
        Needed for :func:`tomo.utils.tomo_output.write_plotinfo_ftn`
        function in order to write the original output format.
    upper bunch limit
        Estimation of the upper bunch limit in bin numbers.
        Needed for :func:`tomo.utils.tomo_output.write_plotinfo_ftn`
        function in order to write the original output format.

    """

    ref_idx = machine.beam_ref_frame
    ref_prof = waterfall[ref_idx]
    ref_turn = ref_idx * machine.dturns

    # Find the upper and lower tangent for of the bunch.
    # Needed in order to calculate the duration [rad] of the bunch.
    tfoot_up, tfoot_low = _calc_tangentfeet(ref_prof)

    # Calculate the duration of the bunch [rad].
    bunch_duration = (tfoot_up - tfoot_low) * machine.dtbin
    bunch_phaselength = (machine.h_num * bunch_duration
                         * machine.omega_rev0[ref_turn])

    # Estimate the synchronous phase.
    x0 = machine.phi0[ref_turn] - bunch_phaselength / 2.0
    phil = optimize.newton(
        func=physics._phase_low, x0=x0,
        fprime=physics._dphase_low,
        tol=0.0001, maxiter=100,
        args=(bunch_phaselength, machine.vrf1_at_turn, machine.vrf2_at_turn,
              machine.phi0, machine.h_ratio, machine.phi12, ref_turn))

    # Calculates the x coordinate of the synchronous particle given in
    # the phase space coordinate system.
    fitted_synch_part_x = (tfoot_low + (machine.phi0[ref_turn] - phil)
                           / (machine.h_num
                              * machine.omega_rev0[ref_turn]
                              * machine.dtbin))

    return fitted_synch_part_x, tfoot_low, tfoot_up


@dispatch(Profiles)
def fit_synch_part_x(profiles: 'Profiles') -> Tuple[np.ndarray, float, float]:
    """Linear fit to estimate the phase coordinate of the synchronous
    particle. The found phase is returned as a x-coordinate of the phase space
    coordinate systems in fractions of bins. The estimation is done at
    the beam reference profile, which is set in the
    :class:`tomo.tracking.machine.Machine` object.

    Parameters
    ----------
    profiles: Profiles
        Profiles object containing waterfall and information about the
        measurement.

    Returns
    -------
    fitted_synch_part_x
        X coordinate in the phase space coordinate system of the synchronous
        particle given in bin numbers.
    lower bunch limit
        Estimation of the lower bunch limit in bin numbers.
        Needed for :func:`tomo.utils.tomo_output.write_plotinfo_ftn`
        function in order to write the original output format.
    upper bunch limit
        Estimation of the upper bunch limit in bin numbers.
        Needed for :func:`tomo.utils.tomo_output.write_plotinfo_ftn`
        function in order to write the original output format.

    """
    return fit_synch_part_x(profiles.waterfall, profiles.machine)


def cut_waterfall(waterfall: Union[List[np.ndarray], np.ndarray],
                  cut_left: int, cut_right: int) -> np.ndarray:
    """
    Cut a waterfall array of shape (n_profiles, n_bins) to shape
    (n_profiles, (cut_right - cut_left)) by shaving off the bins
    to the left of cut_left, and to the right of cut_right

    Parameters
    ----------
    waterfall : List[np.ndarray] or np.npdarray
        2D array of raw-data shaped as waterfall. Shape: (nprofiles, nbins).
    cut_left : int
        The left hand side of the cut, counted in bin number
    cut_right :
        The right hands side of the cut, counted in bin number
        Can be negative, and the bin number count is then used from
        the right side of the array.

    Returns
    -------
    np.ndarray :
        A cut waterfall of shape (n_profiles, (cut_right - cut_left))
    """
    if isinstance(waterfall, list):
        waterfall = np.ndarray(waterfall)

    n_bins = len(waterfall[0])
    asrt.assert_greater_or_equal(cut_left, 'cut_left', 0, ValueError)
    if cut_right < 0:
        asrt.assert_inrange(cut_right, 'cut_right', -n_bins, -1, ValueError)
        asrt.assert_greater(n_bins + cut_right, 'cut_right', cut_left,
                            ValueError,
                            extra_text='-cut_right cannot wrap around'
                                       'cut_left')
    else:
        asrt.assert_inrange(cut_right, 'cut_right', 0, n_bins, ValueError,
                            'cut_right cannot be greater than the number'
                            'of bins in the waterfall')
        asrt.assert_greater(cut_right, 'cut_right', cut_left, ValueError,
                            'cut_right must be greater than cut_left')

    waterfall = waterfall[:, cut_left:cut_right]
    return waterfall


# Finds foot tangents of profile. Needed to estimate bunch duration
# when performing a fit to find synch_part_x.
def _calc_tangentfeet(ref_prof: np.ndarray) -> Tuple[float, float]:
    nbins = len(ref_prof)
    index_array = np.arange(nbins) + 0.5

    tanbin_up, tanbin_low = _calc_tangentbins(ref_prof, nbins)

    [bl, al] = np.polyfit(index_array[tanbin_low - 2: tanbin_low + 2],
                          ref_prof[tanbin_low - 2: tanbin_low + 2], deg=1)

    [bu, au] = np.polyfit(index_array[tanbin_up - 1: tanbin_up + 3],
                          ref_prof[tanbin_up - 1: tanbin_up + 3], deg=1)

    tanfoot_low = -1 * al / bl
    tanfoot_up = -1 * au / bu

    return tanfoot_up, tanfoot_low


# Returns index of last bins to the left and right of max valued bin,
# with value over the threshold.
def _calc_tangentbins(ref_profile: np.ndarray, nbins: int,
                      threshold_coeff: float = 0.15) -> Tuple[float, float]:
    threshold = threshold_coeff * np.max(ref_profile)
    maxbin = np.argmax(ref_profile)
    for ibin in range(maxbin, 0, -1):
        if ref_profile[ibin] < threshold:
            tangent_bin_low = ibin + 1
            break
    for ibin in range(maxbin, nbins):
        if ref_profile[ibin] < threshold:
            tangent_bin_up = ibin - 1
            break

    return tangent_bin_up, tangent_bin_low


# Rebins an 2d array given a rebin factor (rbn).
# The given array MUST have a length equal to an even number.
def _rebin_dividable(data: np.ndarray, rbn: int) -> np.ndarray:
    if data.shape[1] % rbn != 0:
        raise AssertionError('Input array must be '
                             'dividable on the rebin factor.')
    ans = np.copy(data)

    nprofs = data.shape[0]
    nbins = data.shape[1]

    new_nbins = int(nbins / rbn)
    all_bins = new_nbins * nprofs

    ans = ans.reshape((all_bins, rbn))
    ans = np.sum(ans, axis=1)
    ans = ans.reshape((nprofs, new_nbins))

    return ans


# Rebins an 2d array given a rebin factor (rbn).
# The given array MUST have vector length equal to an odd number.
def _rebin_individable(data: np.ndarray, rbn: int) -> np.ndarray:
    nprofs = data.shape[0]
    nbins = data.shape[1]

    ans = np.zeros((nprofs, int(nbins / rbn) + 1))

    last_data_idx = int(nbins / rbn) * rbn
    ans[:, :-1] = _rebin_dividable(data[:, :last_data_idx], rbn)
    ans[:, -1] = _rebin_last(data, rbn)[:, 0]
    return ans


# Rebins last indices of an 2d array given a rebin factor (rbn).
# Needed for the rebinning of odd arrays.
def _rebin_last(data: np.ndarray, rbn: int) -> np.ndarray:
    nprofs = data.shape[0]
    nbins = data.shape[1]

    i0 = (int(nbins / rbn) - 1) * rbn
    ans = np.copy(data[:, i0:])
    ans = np.sum(ans, axis=1)
    ans[:] *= rbn / (nbins - i0)
    ans = ans.reshape((nprofs, 1))
    return ans
