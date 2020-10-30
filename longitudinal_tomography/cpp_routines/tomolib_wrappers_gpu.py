"""Module containing Python wrappers for C++ functions.

Should only be used by advanced users.

:Author(s): **Christoffer Hjertø Grindheim**
"""

import ctypes as ct
from glob import glob
import logging
import os
import sys
from typing import Tuple, TYPE_CHECKING

import numpy as np

from ..utils import exceptions as expt

if TYPE_CHECKING:
    from ..tracking.machine import Machine

log = logging.getLogger(__name__)

_tomolib_pth = os.path.dirname(os.path.realpath(__file__))

# Setting system specific parameters
# NB: the wildcard is to deal with how setup.py names compiled libraries
if 'posix' in os.name:
    _tomolib_pth = glob(os.path.join(_tomolib_pth, 'libgputomo.so'))
elif 'win' in sys.platform:
    _tomolib_pth = glob(os.path.join(_tomolib_pth, 'tomolib*.dll'))
else:
    msg = 'YOU ARE NOT USING A WINDOWS' \
          'OR LINUX OPERATING SYSTEM. ABORTING...'
    raise SystemError(msg)

if len(_tomolib_pth) != 1:
    raise expt.LibraryNotFound('Could not find library. Try reinstalling '
                               'the package with '
                               'python setup.py install.')
_tomolib_pth = _tomolib_pth[0]

# Attempting to load C++ library
if os.path.exists(_tomolib_pth):
    log.debug(f'Loading C++ library: {_tomolib_pth}')
    _tomolib = ct.CDLL(_tomolib_pth)
else:
    error_msg = f'\n\nCould not find library at:\n{_tomolib_pth}\n' \
                f'\n- Try to run compile.py in the tomo directory\n'
    raise expt.LibraryNotFound(error_msg)

# Needed for sending 2D arrays to C++ functions.
# Pointer to pointers.
_double_ptr = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')

# ========================================
#           Setting argument types
# ========================================

# NB! It is critical that the input are of the same data type as specified
#       in the arg types. The correct data types can also be found in the
#       declarations of the C++ functions. Giving arrays of datatype int
#       to a C++ function expecting doubles will lead to mystical (and ugly)
#       errors.

# kick and drift
# ---------------------------------------------
_k_and_d = _tomolib.kick_and_drift
_k_and_d.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
                     np.ctypeslib.ndpointer(ct.c_double),
                     np.ctypeslib.ndpointer(ct.c_double),
                     np.ctypeslib.ndpointer(ct.c_double),
                     np.ctypeslib.ndpointer(ct.c_double),
                     np.ctypeslib.ndpointer(ct.c_double),
                     np.ctypeslib.ndpointer(ct.c_double),
                     np.ctypeslib.ndpointer(ct.c_double),
                     np.ctypeslib.ndpointer(ct.c_double),
                     np.ctypeslib.ndpointer(ct.c_double),
                     ct.c_double,
                     ct.c_double,
                     ct.c_int,
                     ct.c_int,
                     ct.c_int,
                     ct.c_int,
                     ct.c_int,
                     ct.c_bool]
_k_and_d.restypes = None
# ---------------------------------------------


# =============================================================
# Functions for particle tracking
# =============================================================


def kick_and_drift(xp: np.ndarray, yp: np.ndarray,
                   denergy: np.ndarray, dphi: np.ndarray,
                   rfv1: np.ndarray, rfv2: np.ndarray, rec_prof: int,
                   nturns: int, nparts: int, *args, machine: 'Machine' = None,
                   ftn_out: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper for full kick and drift algorithm written in C++.

    Tracks all particles from the time frame to be recreated,
    trough all machine turns.

    Used in the :mod:`tomo.tracking.tracking` module.

    Parameters
    ----------
    xp: ndarray
        2D array large enough to hold the phase of each particle
        at every time frame. Shape: (nprofiles, nparts)
    yp: ndarray
        2D array large enough to hold the energy of each particle
        at every time frame. Shape: (nprofiles, nparts)
    denergy: ndarray
        1D array holding the energy difference relative to the synchronous
        particle for each particle the initial turn.
    dphi: ndarray
        1D array holding the phase difference relative to the synchronous
        particle for each particle the initial turn.
    rfv1: ndarray
        Array holding the radio frequency voltage at RF station 1 for each
        turn, multiplied with the charge state of the particles.
    rfv2: ndarray
        Array holding the radio frequency voltage at RF station 2 for each
        turn, multiplied with the charge state of the particles.
    rec_prof: int
        Index of profile to be reconstructed.
    nturns: int
        Total number of machine turns.
    nparts: int
        The number of particles.
    args: tuple
        Arguments can be provided via the args if a machine object is not to
        be used. In this case, the args should be:

        - phi0
        - deltaE0
        - omega_rev0
        - drift_coef
        - phi12
        - h_ratio
        - dturns

        The args will not be used if a Machine object is provided.

    machine: Machine, optional, default=False
        Object containing machine parameters.
    ftn_out: boolean, optional, default=False
        Flag to enable printing of status of tracking to stdout.
        The format will be similar to the Fortran version.
        Note that the **information regarding lost particles
        are not valid**.

    Returns
    -------
    xp: ndarray
        2D array holding every particles coordinates in phase [rad]
        at every time frame. Shape: (nprofiles, nparts)
    yp: ndarray
        2D array holding every particles coordinates in energy [eV]
        at every time frame. Shape: (nprofiles, nparts)
    """
    xp = np.ascontiguousarray(xp.astype(np.float64))
    yp = np.ascontiguousarray(yp.astype(np.float64))

    xp = np.ascontiguousarray(xp.flatten())
    yp = np.ascontiguousarray(yp.flatten())

    denergy = np.ascontiguousarray(denergy.astype(np.float64))
    dphi = np.ascontiguousarray(dphi.astype(np.float64))

    track_args = [xp, yp,
                  denergy, dphi, rfv1.astype(np.float64),
                  rfv2.astype(np.float64)]

    if machine is not None:
        track_args += [machine.phi0, machine.deltaE0, machine.omega_rev0,
                       machine.drift_coef, machine.phi12, machine.h_ratio,
                       machine.dturns]
    elif len(args) == 7:
        # TODO: this should probably be switched to kwargs to increase
        # robustness
        track_args += args
    else:
        raise expt.InputError(
            'Wrong amount of arguments.\n'
            '*args are: phi0, deltaE0, omega_rev0, '
            'drift_coef, phi12, h_ratio, dturns')

    track_args += [rec_prof, nturns, nparts, machine.nprofiles, ftn_out]

    _k_and_d(*track_args)

    xp = xp.reshape((machine.nprofiles, nparts))
    yp = yp.reshape((machine.nprofiles, nparts))
    return xp, yp



# =============================================================
# Utilities
# =============================================================

# Retrieve pointer of ndarray
def _get_pointer(x):
    return x.ctypes.data_as(ct.c_void_p)


# Retrieve 2D pointer.
# Needed for passing two-dimensional arrays to the C++ functions
# as pointers to pointers.
def _get_2d_pointer(arr2d):
    return (arr2d.__array_interface__['data'][0]
            + np.arange(arr2d.shape[0]) * arr2d.strides[0]).astype(np.uintp)
