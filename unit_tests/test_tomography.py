import unittest
import numpy as np
import numpy.testing as nptest
from tomo.reconstruct_c import ReconstructCpp
from tomo.map_info import MapInfo
from tomo.time_space import TimeSpace
from tomo.tomography import Tomography
from tomo.parameters import Parameters
from unit_tests.C500values import C500


class TestTomography(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c500 = C500()
        cls.rec_vals = cls.c500.get_reconstruction_values()
        cls.tomo_vals = cls.c500.get_tomography_values()

    def test_backproject(self):
        ca = TestTomography.c500.arrays
        cv = TestTomography.c500.values
        tv = TestTomography.tomo_vals
        rv = TestTomography.rec_vals
        ppath = TestTomography.c500.path + "profiles.dat"
        phase_space = np.zeros((cv["reb_profile_length"],
                                cv["reb_profile_length"]))
        Tomography.backproject(np.genfromtxt(ppath),  # TODO: check out
                               phase_space,
                               ca["imin"][0],
                               ca["imax"][0],
                               ca["jmin"],
                               ca["jmax"],
                               rv["maps"],
                               rv["mapsi"],
                               rv["mapsw"],
                               rv["rweights"],
                               rv["fmlistlength"],
                               cv["profile_count"],
                               cv["snpt"])
        nptest.assert_almost_equal(phase_space, tv["first_backproj"],
                                   err_msg="Error in backprojection")

    def test_project(self):
        ca = TestTomography.c500.arrays
        cv = TestTomography.c500.values
        tv = TestTomography.tomo_vals
        rv = TestTomography.rec_vals
        ppath = TestTomography.c500.path + "profiles.dat"

        diffprofiles = (np.genfromtxt(ppath)  # TODO: check out
                        - Tomography.project(
                            tv["first_backproj"],
                            ca["imin"][0],
                            ca["imax"][0],
                            ca["jmin"],
                            ca["jmax"],
                            rv["maps"],
                            rv["mapsi"],
                            rv["mapsw"],
                            rv["fmlistlength"],
                            cv["snpt"],
                            cv["profile_count"],
                            cv["reb_profile_length"]))
        nptest.assert_almost_equal(diffprofiles, tv["first_dproj"],
                                   err_msg="")

    def test_discrepancy(self):
        cv = TestTomography.c500.values
        tv = TestTomography.tomo_vals

        diff = Tomography.discrepancy(tv["first_dproj"],
                                      cv["reb_profile_length"],
                                      cv["profile_count"])
        self.assertAlmostEqual(diff, 0.0011512802609203203,
                               msg="Error in calculation of "
                                   "discrepancy")

    def test_supress_and_norm(self):
        tv = TestTomography.tomo_vals
        phase_space = tv["ps_before_norm"]

        phase_space = Tomography.supress_zeroes_and_normalize(phase_space)
        nptest.assert_almost_equal(phase_space, tv["ps_after_norm"],
                                   err_msg="Error in suppression and "
                                           "normalization of phase space")

    def test_surpress_zeroes_bad_input(self):
        phase_space = np.zeros((100, 100))
        phase_space[51] = -1
        with self.assertRaises(Exception):
            _ = Tomography.supress_zeroes_and_normalize(phase_space)

    def test_bad_reconstruction_input(self):
        rec = ReconstructCpp.__new__(ReconstructCpp)
        rec.mapinfo = MapInfo.__new__(MapInfo)
        rec.timespace = TimeSpace.__new__(TimeSpace)
        rec.timespace.par = Parameters.__new__(Parameters)

        rec.timespace.par.profile_count = 100
        rec.timespace.par.profile_length = 150
        rec.mapsi = np.zeros((100, 10))
        rec.mapweights = np.zeros((111, 10))

        test_tomo = Tomography(rec)

        # Testing mapsi and mapweights of unequal lengths
        with self.assertRaises(Exception):
            test_tomo.validate_reconstruction(0)
        rec.mapweights = np.zeros((100, 10))
        rec.maps = np.zeros((19, 150, 150))

        # Testing maps array with unexpected value
        with self.assertRaises(Exception):
            test_tomo.validate_reconstruction(0)
        rec.maps = np.zeros((19, 100, 150))

        # Bad profile to reconstruct argument
        with self.assertRaises(Exception):
            test_tomo.validate_reconstruction(-1)


if __name__ == '__main__':
    unittest.main()
