import logging as log
import numpy as np
from scipy import signal
from physics import e_UNIT, calc_self_field_coeffs
from utils.assertions import assert_equal, assert_inrange, assert_greater
from utils.exceptions import (RawDataImportError, InputError, RebinningError,
                              FilteredProfilesError, ProfilesReducedToZero)
# ================
# About TimeSpace:   <- fix!
# ================
# The TimeSpace class handles import and processing of data in time domain.
#  - importing raw data, and converting it to profiles, specified by input parameters.
#  - calculation of total charge in profile
#  - if xat0, from input parameters, is less than 0 wil a fit be performed to find xat0
#  - calculations of yat0, phi wrap, xorigin
#  - if self field flag is true will self fields be included in the reconstruction.
#       in the time space class will a savitzky-golay smoothing filter be applied to the profiles
#       and the self field voltage will be calculated.
#
# Variables used in this class are retrieved from a parameter object,
#   which stores the input parameters for the reconstruction.
#   A description of all the input and time space variables can be found in the parameters module.
#
# ====================
# TimeSpace variables:
# ====================
# par               parameter object (for more info, see parameters module)
# profiles          a (profile-count, profile-length) shaped matrix containing profile data, ready for tomography.
# profile_charge    Total charge in the reference profile
#
# Self field variables:
# ---------------------
# vself             Self-field voltage
# dsprofiles        Smoothed derivative of profiles
#
# bunch_phaselength Bunch phase length in beam reference profile
# tangentfoot_low   Used for estimation of bunch duration
# tangentfoot_up
# phiwrap           Phase covering the an integer of rf periods
# wrap_length       Maximum number of bins to cover an integer number of rf periods
# fitted_xat0          Value of (if) fitted xat0


class Profiles:

    def __init__(self, machine, sampling_time, waterfall=None):
        self.machine = machine
        self.sampling_time = sampling_time
        if waterfall is not None:
            self.waterfall = waterfall

    @property
    def waterfall(self):
        return self._waterfall

    @waterfall.setter
    def waterfall(self, new_waterfall):
        if not hasattr(new_waterfall, '__iter__'):
            raise InputError('waterfall should be an iterable.')
        log.info('Loading waterfall...')
        self._waterfall = new_waterfall.clip(0.0)
        if np.sum(np.abs(self.waterfall)) == 0.0:
            raise ProfilesReducedToZero()
        self.profile_charge = self.calc_profilecharge(
                                self._waterfall[self.machine.beam_ref_frame])
        self._waterfall = (self._waterfall
                            / np.vstack(np.sum(self._waterfall, axis=1)))
        self.machine.nbins = self._waterfall.shape[1]
        log.info(f'Waterfall loaded. (profile charge: '
                 f'{self.profile_charge:.3E}, nbins: {self.machine.nbins})')

    # Calculate the total charge of profile
    def calc_profilecharge(self, ref_prof):
        return (np.sum(ref_prof) * self.sampling_time
                / (e_UNIT * self.machine.pickup_sensitivity))

    # Calculate self-fields based on filtered profiles.
    # If filtered profiles are not provided by the user,
    # standard filter (savitzky-golay smoothing filter) is used.
    def calc_self_fields(self, filtered_profiles=None):
        if filtered_profiles is None:
            self.dsprofiles = signal.savgol_filter(
                                x=self._waterfall, window_length=7,
                                polyorder=4, deriv=1)
        else:
            self.dsprofiles = self._check_manual_filtered_profs(
                                                    filtered_profiles)

        (self.phiwrap,
         self.wrap_length) = self._find_wrap_length()

        self.vself = self._calculate_self()
        log.info('Self fields were calculated.')

    def _check_manual_filtered_profs(self, fprofs):
        if not hasattr(fprofs, '__iter__'):
            err_msg = 'Filtered profiles should be iterable.'
            raise FilteredProfilesError(err_msg)
        fprofs = np.array(fprofs)
        if fprofs.shape == self.waterfall.shape:
            return fprofs
        else:
            err_msg = f'Filtered profiles should have the same shape'\
                      f'as the waterfall.\n'\
                      f'Shape profiles: {fprofs.shape}\n'\
                      f'Shape waterfall: {self.waterfall.shape}\n'
            raise FilteredProfilesError(err_msg)

    # Calculate the number of bins in the first
    # integer number of rf periods, larger than the image width.
    def _find_wrap_length(self):
        if self.machine.bdot > 0.0:
            last_turn_index = ((self.machine.nprofiles - 1)
                               * self.machine.dturns - 1)
            drad_bin = (self.machine.h_num
                        * self.machine.omega_rev0[last_turn_index]
                        * self.machine.dtbin)
        else:
            drad_bin = (self.machine.h_num
                        * self.machine.omega_rev0[0]
                        * self.machine.dtbin)

        phiwrap = np.ceil(self.machine.nbins * drad_bin
                          / (2 * np.pi)) * 2 * np.pi

        wrap_length = int(np.ceil(phiwrap / drad_bin))

        log.info(f'phi wrap =  {str(phiwrap)}, '
                 f'wrap length =  {str(wrap_length)}')
        return phiwrap, wrap_length

    # Calculate self-field voltages
    def _calculate_self(self):
        sfc = calc_self_field_coeffs(self.machine)
        vself = np.zeros((self.machine.nprofiles - 1,
                          self.wrap_length + 1),
                         dtype=float)
        for i in range(self.machine.nprofiles - 1):
            vself[i, :self.machine.nbins] = (0.5 * self.profile_charge
                                             * (sfc[i]
                                             * self.dsprofiles[
                                                i,:self.machine.nbins]
                                             + sfc[i + 1]
                                             * self.dsprofiles[
                                                i + 1, :self.machine.nbins]))
        return vself
