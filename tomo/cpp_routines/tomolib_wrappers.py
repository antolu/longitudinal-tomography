'''Module containing Python wrappers for C++ functions.

:Author(s): **Christoffer Hjertø Grindheim**
'''

import ctypes as ct
import logging as log
import numpy as np
import os
import sys

from ..utils import exceptions as expt


_tomolib_pth = os.path.dirname(os.path.realpath(__file__))

# Setting system spescific parameters
if 'posix' in os.name:
    _tomolib_pth = os.path.join(_tomolib_pth, 'tomolib.so')
elif 'win' in sys.platform:
    _tomolib_pth = os.path.join(_tomolib_pth, 'tomolib.dll')
else:
    msg = 'YOU ARE NOT USING A WINDOWS'\
          'OR LINUX OPERATING SYSTEM. ABORTING...'
    raise SystemError(msg)

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

# Kick and drift (gpu version)
# _k_and_d_gpu = _tomolib.kick_and_drift_gpu
# _k_and_d_gpu.argtypes = _k_and_d.argtypes

# ========================================
#           Setting argument types
# ========================================

# kick and drift
# ---------------------------------------------
_k_and_d = _tomolib.kick_and_drift
_k_and_d.argtypes = [_double_ptr,
                     _double_ptr,
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
                     ct.c_bool]
_k_and_d.restypes = None
# ---------------------------------------------

# Reconstruction routine (flat version)
_reconstruct = _tomolib.reconstruct
_reconstruct.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
                         _double_ptr,
                         np.ctypeslib.ndpointer(ct.c_double),
                         np.ctypeslib.ndpointer(ct.c_double),
                         ct.c_int, ct.c_int,
                         ct.c_int, ct.c_int,
                         ct.c_bool]

# Back_project (flat version)
_back_project = _tomolib.back_project
_back_project.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
                          _double_ptr, np.ctypeslib.ndpointer(ct.c_double)]
_back_project.restypes = None

# Project (flat version)
_proj = _tomolib.project
_proj.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
                  _double_ptr, np.ctypeslib.ndpointer(ct.c_double)]
_proj.restypes = None


# =============================================================
# Functions for paricle tracking
# =============================================================


def kick(machine, denergy, dphi, rfv1, rfv2, npart, turn, up=True):
    '''Wrapper for C++ kick function.

    Particle kick for **one** machine turn.

    Parameters
    ----------
    machine: Machine
        Object holding machine parameters.
    denergy: ndarray
        Array holding the energy difference relative to the synchronous
        particle for each particle at a turn.
    dphi: ndarray
        Array holding the phase difference relative to the synchronous
        particle for each particle at a turn.
    rfv1: ndarray
        Array holding the radio frequency voltage
        at RF station 1 for each turn, multiplied with the
        charge state of the particles.
    rfv2: ndarray
        Array holding the radio frequency voltage
        at RF station 2 for each turn, multiplied with the
        charge state of the particles.
    npart: int
        Number of tracked particles.
    turn: int
        Current machine turn.
    up: boolean
        Direction of tracking. Up=True tracks towards last machine turn,
        up=False tracks toward first machine turn.

    Returns
    -------
    denergy: ndarray
        The new energy of each particle after voltage kick.
    '''
    args = (_get_pointer(dphi),
            _get_pointer(denergy),
            ct.c_double(rfv1[turn]),
            ct.c_double(rfv2[turn]),
            ct.c_double(machine.phi0[turn]),
            ct.c_double(machine.phi12),
            ct.c_double(machine.h_ratio),
            ct.c_int(npart),
            ct.c_double(machine.deltaE0[turn]))
    if up:
        _tomolib.kick_up(*args)
    else:
        _tomolib.kick_down(*args)
    return denergy


def drift(denergy, dphi, drift_coef, npart, turn, up=True):
    '''Wrapper for C++ drift function.

    Particle drift for **one** machine turn

    Parameters
    ----------
    denergy: ndarray
        Array holding the energy difference relative to the synchronous
        particle for each particle at a turn.
    dphi: ndarray
        Array holding the phase difference relative to the synchronous
        particle for each particle at a turn.
    drift_coef: ndarray
        Array of drift coefficient at each machine turn
    npart: int
        Number of tracked particles.
    turn: int
        Current machine turn.
    up: boolean
        Direction of tracking. Up=True tracks towards last machine turn,
        up=False tracks toward first machine turn.

    Returns
    -------
    dphi: ndarray
        The new phase for each particle after drifting for a machine turn.
    '''
    args = (_get_pointer(dphi),
            _get_pointer(denergy),
            ct.c_double(drift_coef[turn]),
            ct.c_int(npart))
    if up:
        _tomolib.drift_up(*args)
    else:
        _tomolib.drift_down(*args)
    return dphi


def kick_and_drift(xp, yp, denergy, dphi, rfv1, rfv2, rec_prof,
                   nturns, nparts, *args, machine=None, ftn_out=False):
    '''Wrapper for full kick and drift algorithm written in C++.
    
    Tracks all particles from the time frame to be recreated,
    trough all machine turns.

    Parameters
    ----------
    xp: ndarray
        Array large enough to hold the phase of each particle
        at every time frame. (nprofiles, nparts)
    yp: ndarray
        Array large enough to hold the energy of each particle
        at every time frame. (nprofiles, nparts)
    denergy: ndarray
        Array holding the energy difference relative to the synchronous
        particle for each particle at a turn.
    dphi: ndarray
        Array holding the phase difference relative to the synchronous
        particle for each particle at a turn.
    rfv1: ndarray
        Array holding the radio frequency voltage
        at RF station 1 for each turn, multiplied with the
        charge state of the particles.
    rfv2: ndarray
        Array holding the radio frequency voltage
        at RF station 2 for each turn, multiplied with the
        charge state of the particles.
    rec_prof: int
        Index of profile to be reconstructed.
    nturns: int
        Total number of machine turns.
    nparts: int
        Number of particles.
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

    machine: Machine
        Object containing machine parameters.
    ftn_out: boolean
        Flag to enable printing of status of tracking to stdout.
        The format will be similar to the Fortran version.
        Note that the **information regarding lost particles
        are not valid**.
    
    Returns
    -------
    xp: ndarray
        Array holding every particles coordinates in phase [rad]
        at every time frame. 
    yp: ndarray
        Array holding every particles coordinates in energy [eV]
        at every time frame.
    '''
    xp = np.ascontiguousarray(xp.astype(np.float64))
    yp = np.ascontiguousarray(yp.astype(np.float64))

    denergy = np.ascontiguousarray(denergy.astype(np.float64))
    dphi = np.ascontiguousarray(dphi.astype(np.float64))

    track_args = [_get_2d_pointer(xp), _get_2d_pointer(yp),
                  denergy, dphi, rfv1.astype(np.float64),
                  rfv2.astype(np.float64)]

    if machine is not None:
        track_args += [machine.phi0, machine.deltaE0, machine.omega_rev0,
                       machine.drift_coef, machine.phi12, machine.h_ratio,
                       machine.dturns]
    elif len(args) == 7:
        track_args += args
    else:
        raise expt.InputError(
            'Wrong amount of arguments.\n'
            '*args are: phi0, deltaE0, omega_rev0, '
            'drift_coef, phi12, h_ratio, dturns')
    
    track_args += [rec_prof, nturns, nparts, ftn_out]

    _k_and_d(*track_args)
    return xp, yp

# =============================================================
# Functions for phase space reconstruction
# =============================================================


def back_project(weights, flat_points, flat_profiles, nparts, nprofs):
    '''Wrapper for back projection routine written in C++.
    
    Parameters
    ----------
    weights: ndarray
        Weight of each particle.
    flat_points: ndarray
        Particle coordinates as integers, pointing at flattened versions
        of the waterfall.
    flat_profiles: ndarray
        Flattened waterfall.
    nparts: int
        Number of tracked particles.
    nprofs: int
        Number of profiles.

    Returns
    -------
    weights: ndarray
        Updated weight of each particle. 
    '''
    _back_project(weights, _get_2d_pointer(flat_points),
                  flat_profiles, nparts, nprofs)
    return weights


def project(recreated, flat_points, weights, nparts, nprofs, nbins):
    '''Wrapper projection routine written in C++.

    Parameters
    ----------
    recreated: ndarray
        Array with the shape of the waterfall
        to be recreated, initiated as zero.
    flat_points: ndarray
        Particle coordinates as integers, pointing at flattened versions
        of the waterfall.
    weights: ndarray
        Weight of each particle.
    nparts: int
        Number of tracked particles.
    nprofs: int
        Number of profiles.
    nbins: int
        Number of bins in profiles.

    Returns
    -------
    recreated: ndarray
        Projected waterfall.
    '''
    recreated = np.ascontiguousarray(recreated.flatten())
    _proj(recreated, _get_2d_pointer(flat_points), weights, nparts, nprofs)
    recreated = recreated.reshape((nprofs, nbins))
    return recreated


def reconstruct(weights, xp, flat_profiles, discr,
                niter, nbins, npart, nprof, verbose):
    '''Wrapper for full reconstruction in C++.

    Parameters
    ----------
    weights: ndarray
        Array containing the weight of each particle initiated to zeroes.
    xp: ndarray
        The coordinate of each particle at each time frame
        (shape: (nparts, nprofiles)). Coordinates should be as integers
        in the phase space coordinate system.
    flat_profiles: ndarray
        Flattened waterfall.
    discr: ndarray
        Array large enough to hold the discrepancy for each iteration of the
        reconstruction process + 1.
    niter: int
        Number of iterations in the reconstruction process.
    nbins: int
        Number of bins in a profile.
    npart: int
        Number of tracked particles.
    nprof: int
        Number of profiles.
    verbose: boolean
        Flag to indicate that the tomography routine should broadcast its
        status to stdout. The output is identical to the output 
        from the Fortran version.

    Returns
    -------
    weights: ndarray
        Final weight of each particle.
    discr: ndarray
        discrepancy at each iteration of the reconstruction.
    '''
    _reconstruct(weights, _get_2d_pointer(xp), flat_profiles,
                 discr, niter, nbins, npart, nprof, verbose)
    return weights, discr


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
