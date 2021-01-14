"""Module containing functions for treatment of data.

:Author(s): **Christoffer Hjertø Grindheim**
"""

from typing import Tuple, TYPE_CHECKING, Union
from warnings import warn

import numpy as np

from tomo import exceptions as expt
from tomo.utils import physics
from . import pre_process

if TYPE_CHECKING:
    from tomo.data.profiles import Profiles
    from tomo.tracking.machine import Machine
    from tomo.tomography.__tomography import Tomography


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
        Otherwise, None will be returned.
    """
#    warn('The rebin function has been moved to tomo.data.pre_process')
    return pre_process.rebin(waterfall, rbn, dtbin, synch_part_x)


# Original function for finding synch_part_x
# Finds synch_part_x based on a linear fit on a reference profile.
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
#    warn('The fit_synch_part_x function has moved to '
#         'tomo.data.pre_process')
    return pre_process.fit_synch_part_x(profiles)


def phase_space(tomo: 'Tomography', machine: 'Machine', profile: int = 0) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """returns time, energy and phase space density arrays from a
    reconstruction, requires the homogenous distribution to have been
    generated by the particles class.

    Parameters
    ----------
    tomo: Tomography
        Object holding the information about a tomographic reconstruction.
    machine: Machine
        Object holding information about machine and reconstruction parameters.
    profile: int
        Index of profile to be reconstructed.

    Returns
    -------
    t_range: ndarray
        1D array containing time axis of reconstructed phase space image.
    E_range: ndarray
        1D array containing energy axis of reconstructed phase space image.
    density: ndarray
        2D array containing the reconstructed phase space image.

    Raises
    ------
    InputError: Exception
        phase_space function requires automatic phase space generation
        to have been used.

    """
    if machine.dEbin is None:
        raise expt.InputError("""phase_space function requires automatic
                              phase space generation to have been used""")

    density = _make_phase_space(tomo.xp[:, profile], tomo.yp[:, profile],
                                tomo.weight, machine.nbins)

    t_cent = machine.synch_part_x
    E_cent = machine.synch_part_y

    t_range = (np.arange(machine.nbins) - t_cent) * machine.dtbin
    E_range = (np.arange(machine.nbins) - E_cent) * machine.dEbin

    return t_range, E_range, density


# Creates a [nbins, nbins] array populated with the weights of each test
# particle
def _make_phase_space(xp: np.ndarray, yp: np.ndarray, weights: np.ndarray,
                      nbins: int) -> np.ndarray:
    phase_space = np.zeros([nbins, nbins])

    for x, y, w in zip(xp, yp, weights):
        phase_space[x, y] += w

    return phase_space


def calc_baseline_ftn(*args):
#    warn('This function has moved to tomo.compat.fortran.')
    from tomo.compat.fortran import calc_baseline
    return calc_baseline(*args)
