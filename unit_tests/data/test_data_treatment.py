"""Unit-tests for the physics module.

Run as python test_data_treatment.py in console or via coverage
"""

import os
import unittest

import numpy as np
import numpy.testing as nptest

from .. import commons
import tomo.data.data_treatment as treat
import tomo.data.profiles as prf
import tomo.tracking.machine as mch

# Machine arguments based on the input file INDIVShavingC325.dat
MACHINE_ARGS = commons.get_machine_args()


class TestDataTreatment(unittest.TestCase):

    def test_rebin_odd_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nbins = 9
        machine.synch_part_x = 4.53
        machine.dtbin = 0.535

        waterfall = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                              [10, 21, 32, 43, 54, 65, 76, 86, 97]],
                             dtype=float)
        rbn = 2
        (rebinned,
         out_dtbin,
         out_sync_pt_x) = treat.rebin(waterfall, rbn, machine.dtbin,
                                      machine.synch_part_x)

        correct_rebinned = np.array([[3.0, 7.0, 11.0, 15.0, 16.0],
                                     [31.0, 75.0, 119.0, 162.0, 172.66666667]])

        nptest.assert_almost_equal(
            rebinned, correct_rebinned,
            err_msg='Rebinning array of odd length failed')

        # Checks that the x coordinate of synchronous particle
        # is updated to fit the new number of bins.
        updated_synch_part_x = 2.265
        self.assertAlmostEqual(out_sync_pt_x, updated_synch_part_x,
                               msg='Error in updated synch part x')

        # Checks that the size of the bins are updated
        # is updated to fit the new number of bins.
        updated_dtbin = 1.07
        self.assertAlmostEqual(out_dtbin, updated_dtbin,
                               msg='Error in updated dtbin')

    def test_rebin_even_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nbins = 8
        machine.synch_part_x = 4.53
        machine.dtbin = 0.535

        waterfall = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                              [10, 21, 32, 43, 54, 65, 76, 86]],
                             dtype=float)
        rbn = 2
        (rebinned,
         out_dtbin,
         out_sync_pt_x) = treat.rebin(waterfall, rbn, machine.dtbin,
                                      machine.synch_part_x)

        correct_rebinned = np.array([[3.0, 7.0, 11.0, 15.0],
                                     [31.0, 75.0, 119.0, 162.0]])

        nptest.assert_almost_equal(
            rebinned, correct_rebinned,
            err_msg='Rebinning array of odd length failed')

        # Checks that the x coordinate of synchronous particle
        # is updated to fit the new number of bins.
        updated_synch_part_x = 2.265
        self.assertAlmostEqual(out_sync_pt_x, updated_synch_part_x,
                               msg='Error in updated synch part x')

        # Checks that the size of the bins are updated
        # is updated to fit the new number of bins.
        updated_dtbin = 1.07
        self.assertAlmostEqual(out_dtbin, updated_dtbin,
                               msg='Error in updated dtbin')

    def test_fit_synch_part_x_correct_x_coord(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.time_at_turn = np.array([1.3701195948153858e-06])
        machine.omega_rev0 = np.array([4585866.32214847])
        machine.phi0 = np.array([0.40078213])

        waterfall = self._load_waterfall()
        profile = prf.Profiles(machine, machine.dtbin, waterfall)

        correct_x = 134.08972093246575
        fitted_x, _, _ = treat.fit_synch_part_x(profile)

        self.assertAlmostEqual(
            fitted_x, correct_x, msg='Fitted x coordinate of synchronous '
                                     'particle is not correct')

    def test_fit_synch_part_x_correct_tfoot_low(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.time_at_turn = np.array([1.3701195948153858e-06])
        machine.omega_rev0 = np.array([4585866.32214847])
        machine.phi0 = np.array([0.40078213])

        waterfall = self._load_waterfall()
        profile = prf.Profiles(machine, machine.dtbin, waterfall)

        correct_tfoot_low = 21.569078947368386
        _, tfoot_low, _ = treat.fit_synch_part_x(profile)

        self.assertAlmostEqual(
            tfoot_low, correct_tfoot_low,
            msg='Lower tangent foot is calculated incorrectly')

    def test_fit_synch_part_x_correct_tfoot_up(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.time_at_turn = np.array([1.3701195948153858e-06])
        machine.omega_rev0 = np.array([4585866.32214847])
        machine.phi0 = np.array([0.40078213])

        waterfall = self._load_waterfall()
        profile = prf.Profiles(machine, machine.dtbin, waterfall)

        correct_tfoot_up = 255.84868421052698
        _, _, tfoot_up = treat.fit_synch_part_x(profile)

        self.assertAlmostEqual(
            tfoot_up, correct_tfoot_up,
            msg='Upper tangent foot is calculated incorrectly')

    def _load_waterfall(self):
        base_dir = os.path.split(os.path.realpath(__file__))[0]
        base_dir = os.path.split(base_dir)[0]
        data_path = os.path.join(base_dir, 'resources')

        waterfall = np.load(os.path.join(
            data_path, 'waterfall_INDIVShavingC325.npy'))
        return waterfall
