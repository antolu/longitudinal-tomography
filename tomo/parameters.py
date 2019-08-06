# TEMP
import sys
import time as tm
import pandas as pd
# END TEMP
import logging
import physics
import numpy as np
import numeric
from utils.assertions import TomoAssertions as ta
from utils.exceptions import (InputError,
							  MachineParameterError,
							  SpaceChargeParameterError)

# ================
# About the class:
# ================
# The parameter class receives input parameters and machine settings for the reconstruction.
# These parameters creates the basis of the interpretation of the measured profiles in the time_space class,
# and the particle tracking in the reconstruction class. The parameter class also stores
# information created in time space calculations.
#
# =====================================
# Parameters to be collected from file:
# =====================================
# Settings for reconstruction:
# ----------------------------
# xat0              Synchronous phase in bins in beam_ref_frame
#                       read form file as time (in frame bins) from the lower profile bound to the synchronous phase
#                       (if < 0, a fit is performed) in the "bunch reference" frame
# yat0              Synchronous energy (0 in relative terms) in reconstructed phase space coordinate system
# rebin             Re-binning factor - Number of frame bins to re-bin into one profile bin
# rawdata_file      Input data file
# output_dir        Directory in which to write all output
# framecount        Number of frames in input data
# framelength       Length of each trace in the 'raw' input file - number of bins in each frame
# dtbin             Pixel width (in seconds)
# demax             maximum energy of reconstructed phase space
# dturns			Number of machine turns between each measurement
# preskip_length    Subtract this number of bins from the beginning of the 'raw' input traces
# postskip_length   Subtract this number of bins from the end of the 'raw' input traces
# imin_skip         Number of frame bins after the lower profile bound to treat as empty during reconstruction
# imax_skip         Number of frame bins after the upper profile bound to treat as empty during reconstruction
# framecount        Number of frames in input data
# frame_skipcount   Ignore this number of frames/traces from the beginning of the 'raw' input file
# snpt			    Square root of number of test particles tracked from each pixel of reconstructed phase space
# num_iter  		Number of iterations in the reconstruction process
# machine_ref_frame Frame to which machine parameters are referenced (b0,VRF1,VRF2...)
# beam_ref_frame    Frame to which beam parameters are referenced
# filmstart         First profile to be reconstructed
# filmstop          Last profile to be reconstructed
# filmstep          Step between consecutive reconstructions for the profiles from filmstart to filmstop
# full_pp_flag      If set, all pixels in reconstructed phase space will be tracked
#
# Machine and Particle Parameters:
# --------------------------------
# vrf1, vrf2        Peak voltage of first and second RF system at machine_ref_frame
# vrfXdot           Time derivatives of the RF voltages (considered constant)
# mean_orbit_rad    Machine mean orbit radius (in m)
# bending_rad       Machine bending radius    (in m)
# b0                B-field at machine_ref_frame
# bdot              Time derivative of B-field (considered constant)
# phi12             Phase difference between the two RF systems (considered constant)
# h_ratio           Ratio of harmonics between the two RF systems
# h_num             Principle harmonic number
# trans_gamma       Transitional gamma
# e_rest            Rest energy of accelerated particle
# q                 Charge state of accelerated particle
#
# Space charge parameters:
# ------------------------
# self_field_flag       Flag to include self-fields in the tracking
# g_coupling            Space charge coupling coefficient (geometrical coupling coefficient)
# zwall_over_n          Magnitude of Zwall/n, reactive impedance (in Ohms per mode number) over a machine turn
# pickup_sensitivity    Effective pick-up sensitivity (in digitizer units per instantaneous Amp)
#
# ======================
# calculated parameters:
# ======================
# Arrays with information for each machine turn:
# ----------------------------------------------
# time_at_turn      Time at each turn, relative to machine_ref_frame at the end of each turn
# omega_rev0        Revolution frequency at each turn
# phi0              Synchronous phase angle at the end of each turn
# dphase            Coefficient used for calculating difference from phase n to phase n + 1.
#                       needed in trajectory height calculator and longtrack.
# sfc               Self-field_coefficient
# beta0             Lorenz beta factor (v/c) at the end of each turn
# eta0              Phase slip factor at each turn
# e0                Total energy of synchronous particle at the end of each turn
# deltaE0           Difference between total energy at the end of the turns n and n-1
#
# calculated variables:
# ---------------------
# profile_count     Number of profiles
# profile_length    Length of profile (in number of bins)
# profile_mini      Index of first and last "active" index in profile
# profile_maxi
# all_data          Total number of data points in the 'raw' input file
#
# Beam reference profile parameters (calculated in TimeSpace):
# -------------------------------------------------------------
# bunch_phaselength     Bunch phase length in beam reference profile
# tangentfoot_low       Used for estimation of bunch duration
# tangentfoot_up
# phiwrap
# wrap_length
# fit_xat0              Value of (if) fitted xat0
# x_origin = 0.0        absolute difference in bins between phase=0
#                           and origin of the reconstructed phase-space coordinate system.


class Parameters:

	def __init__(self):

		# Parameters to be collected from file:
		# =====================================
		self.xat0 = 0.0
		self.yat0 = 0.0

		self.rebin = 0
		self.rawdata_file = ""
		self.output_dir = ""
		self.framecount = 0
		self.framelength = 0
		self.dtbin = 0.0
		self.demax = 0.0
		self.dturns = 0
		self.preskip_length = 0
		self.postskip_length = 0

		self.imin_skip = 0
		self.imax_skip = 0
		self.frame_skipcount = 0
		self.snpt = 0
		self.num_iter = 0
		self.machine_ref_frame = 0
		self.beam_ref_frame = 0
		self.filmstep = 0
		self.filmstart = 0
		self.filmstop = 0
		self.full_pp_flag = False

		# Machine and Particle Parameters:
		self.vrf1 = 0.0
		self.vrf2 = 0.0
		self.vrf1dot = 0.0
		self.vrf2dot = 0.0
		self.mean_orbit_rad = 0.0
		self.bending_rad = 0.0
		self.b0 = 0.0
		self.bdot = 0.0
		self.phi12 = 0.0
		self.h_ratio = 0.0
		self.h_num = 0.0
		self.trans_gamma = 0.0
		self.e_rest = 0.0
		self.q = 0.0

		# Space charge parameters:
		self.self_field_flag = False
		self.g_coupling = 0.0
		self.zwall_over_n = 0.0
		self.pickup_sensitivity = 0.0

		# calculated parameters:
		# ======================
		self.time_at_turn = []
		self.omega_rev0 = []
		self.phi0 = []
		self.dphase = []
		self.deltaE0 = []
		self.sfc = []
		self.beta0 = []
		self.eta0 = []
		self.e0 = []

		self.profile_count = 0
		self.profile_length = 0

		self.profile_mini = 0
		self.profile_maxi = 0

		self.all_data = 0

		# Beam reference profile parameters (timeSpaceOutput):
		self.bunch_phaselength = 0.0
		self.tangentfoot_low = 0.0
		self.tangentfoot_up = 0.0
		self.phiwrap = 0.0
		self.wrap_length = 0
		self.fit_xat0 = 0.0
		self.x_origin = 0.0

	# Calculates parameters based on text file as input
	def get_parameters_txt(self, file_name):
		self._read_txt_input(file_name)
		self._assert_input()
		self._init_parameters()
		self._assert_parameters()

	# Reading parameters from text-file input
	def _read_txt_input(self, file_name):
		length_par_file = 98
		# data = [None] * length_par_file
		data = []

		with open(file_name, 'r') as f:
			for i in range(length_par_file):
				data.append(f.readline().strip('\r\n')) # .strip('\n')

		self.rawdata_file = data[12]
		self.output_dir = data[14]
		self.framecount = int(data[16])
		self.frame_skipcount = int(data[18])
		self.framelength = int(data[20])
		self.dtbin = float(data[22])
		self.dturns = int(data[24])
		self.preskip_length = int(data[26])
		self.postskip_length = int(data[28])
		self.imin_skip = int(data[31])
		self.imax_skip = int(data[34])
		self.rebin = int(data[36])
		self.xat0 = float(data[39])
		self.demax = float(data[41])
		self.filmstart = int(data[43])
		self.filmstop = int(data[45])
		self.filmstep = int(data[47])
		self.num_iter = int(data[49])
		self.snpt = int(data[51])
		self.full_pp_flag = bool(int(data[53]))
		self.beam_ref_frame = int(data[55])
		self.machine_ref_frame = int(data[57])
		self.vrf1 = float(data[61])
		self.vrf1dot = float(data[63])
		self.vrf2 = float(data[65])
		self.vrf2dot = float(data[67])
		self.h_num = float(data[69])
		self.h_ratio = float(data[71])
		self.phi12 = float(data[73])
		self.b0 = float(data[75])
		self.bdot = float(data[77])
		self.mean_orbit_rad = float(data[79])
		self.bending_rad = float(data[81])
		self.trans_gamma = float(data[83])
		self.e_rest = float(data[85])
		self.q = float(data[87])
		self.self_field_flag = bool(int(data[91]))
		self.g_coupling = float(data[93])
		self.zwall_over_n = float(data[95])
		self.pickup_sensitivity = float(data[97])
		logging.info("Read successful from file: " + file_name)


	# Subroutine for setting up parameters based on given input file.
	# Values are calculated immediately after the 'single' cavity of the ring
	def _init_parameters(self):
		self._calc_parameter_arrays()

		# Changes due to re-bin factor > 1
		self.dtbin = self.dtbin * self.rebin
		self.xat0 = self.xat0 / float(self.rebin)

		self.profile_count = self.framecount - self.frame_skipcount
		self.profile_length = (self.framelength - self.preskip_length
							   - self.postskip_length)

		self.profile_mini, self.profile_maxi = self._find_imin_imax()

		# calculating total number of data points in the input file
		self.all_data = self.framecount * self.framelength

		# Find self field coefficient for each profile
		self.sfc = physics.calc_self_field_coeffs(self)

	# Initiating arrays in order to store information about parameters
	# that has a different value every turn.
	def _init_arrays(self, all_turns):
		array_length = all_turns + 1
		self.time_at_turn = np.zeros(array_length)
		self.omega_rev0 = np.zeros(array_length)
		self.phi0 = np.zeros(array_length)
		self.dphase = np.zeros(array_length)
		self.deltaE0 = np.zeros(array_length)
		self.beta0 = np.zeros(array_length)
		self.eta0 = np.zeros(array_length)
		self.e0 = np.zeros(array_length)

	# Calculating start-values for the parameters that changes for each turn.
	# The reference frame where the start-values are calculated is the machine reference frame.
	# (machine ref. frame -1 to adjust for fortran input files)
	def _array_initial_values(self):
		i0 = (self.machine_ref_frame - 1) * self.dturns
		self.time_at_turn[i0] = 0
		self.e0[i0] = physics.b_to_e(self)
		self.beta0[i0] = physics.lorenz_beta(self, i0)
		phi_lower, phi_upper = physics.find_phi_lower_upper(self, i0)
		# Synchronous phase of a particle on the nominal orbit
		self.phi0[i0] = physics.find_synch_phase(self, i0, phi_lower,
												 phi_upper)
		return i0

	# Calculating values that changes for each m. turn.
	# First is the arrays inited at index of machine ref. frame (i0).
	# Based on this value are the rest of the values calculated;
	# first, upwards from i0 to total number of turns + 1, then downwards from i0 to 0 (first turn).
	def _calc_parameter_arrays(self):
		all_turns = self._calc_number_of_turns()
		self._init_arrays(all_turns)
		i0 = self._array_initial_values()

		for i in range(i0 + 1, all_turns + 1):
			self.time_at_turn[i] = (self.time_at_turn[i - 1]
									+ 2 * np.pi * self.mean_orbit_rad
									/ (self.beta0[i - 1] * physics.C))

			self.phi0[i] = numeric.newton(physics.rf_voltage,
										  physics.drf_voltage,
										  self.phi0[i - 1], self, i, 0.001)

			self.e0[i] = (self.e0[i - 1]
						  + self.q
						  * physics.short_rf_voltage_formula(
								self.phi0[i], self.vrf1, self.vrf1dot,
								self.vrf2, self.vrf2dot, self.h_ratio,
								self.phi12, self.time_at_turn, i))

			self.beta0[i] = np.sqrt(1.0 - (self.e_rest/float(self.e0[i]))**2)
			self.deltaE0[i] = self.e0[i] - self.e0[i - 1]
		for i in range(i0 - 1, 0, -1):
			self.e0[i] = (self.e0[i + i]
						  - self.q
						  * physics.short_rf_voltage_formula(
								self.phi0[i + 1], self.vrf1, self.vrf1dot,
								self.vrf2, self.vrf2dot, self.h_ratio,
								self.phi12, self.time_at_turn, i + 1))

			self.beta0[i] = np.sqrt(1.0 - (self.e_rest/self.e0[i])**2)
			self.deltaE0[i] = self.e0[i + 1] - self.e0[i]

			self.time_at_turn[i] = (self.time_at_turn[i + 1]
									- 2 * np.pi * self.mean_orbit_rad
									/ (self.beta0[i] * physics.C))

			self.phi0[i] = numeric.newton(physics.rf_voltage,
										  physics.drf_voltage,
										  self.phi0[i + 1], self, i, 0.001)

		# Calculate phase slip factor at each turn
		self.eta0 = physics.phase_slip_factor(self)

		# Calculates dphase for each turn
		self.dphase = physics.find_dphase(self)

		# Calculate revolution frequency at each turn
		self.omega_rev0 = physics.revolution_freq(self)

	# Finding min and max index in profiles.
	# The indexes outside of these are treated as 0
	def _find_imin_imax(self):
		profile_mini = self.imin_skip/self.rebin
		if (self.profile_length - self.imax_skip) % self.rebin == 0:
			profile_maxi = ((self.profile_length - self.imax_skip)
							/ self.rebin)
		else:
			profile_maxi = ((self.profile_length - self.imax_skip)
							/ self.rebin + 1)
		return int(profile_mini), int(profile_maxi)

	# Asserting that the input parameters from user are valid
	def _assert_input(self):
		# Note that some of the assertions is setting the lower limit as 1.
		# This is because of calibrating from input files meant for Fortran,
		#    where arrays by default starts from 1, to the Python version
		#    with arrays starting from 0.

		# Frame assertions
		ta.assert_greater(self.framecount, "frame count", 0, InputError)
		ta.assert_inrange(self.frame_skipcount, "frame skip-count",
						  0, self.framecount, InputError)
		ta.assert_greater(self.framelength, "frame length", 0, InputError)
		ta.assert_inrange(self.preskip_length, "pre-skip length",
						  0, self.framelength, InputError)
		ta.assert_inrange(self.postskip_length, "post-skip length",
						  0, self.framelength, InputError)

		# Bin assertions
		ta.assert_greater(self.dtbin, "dtbin", 0, InputError,
						  'NB: dtbin is the difference of time in bin')
		ta.assert_greater(self.dturns, "dturns", 0, InputError,
						  'NB: dturns is the number of machine turns'
						  'between each measurement')
		ta.assert_inrange(self.imin_skip, 'imin skip',
						  0, self.framelength, InputError)
		ta.assert_inrange(self.imax_skip, 'imax skip',
						  0, self.framelength, InputError)
		ta.assert_greater_or_equal(self.rebin, 're-binning factor',
								   1, InputError)

		# Assertions: profile to be reconstructed
		ta.assert_greater_or_equal(self.filmstart, 'film start',
								   1, InputError)
		ta.assert_greater_or_equal(self.filmstop, 'film stop',
								   self.filmstart, InputError)
		ta.assert_less_or_equal(abs(self.filmstep), 'film step',
								abs(self.filmstop - self.filmstart + 1),
								InputError)
		ta.assert_not_equal(self.filmstep, 'film step', 0, InputError)

		# Reconstruction parameter assertions
		ta.assert_greater(self.num_iter, 'num_iter', 0, InputError,
						  'NB: num_iter is the number of iterations of the '
						  'reconstruction process')
		ta.assert_greater(self.snpt, 'snpt', 0, InputError,
						  'NB: snpt is the square root '
						  'of #tracked particles.')

		# Reference frame assertions
		ta.assert_greater_or_equal(self.machine_ref_frame,
								   'machine ref. frame',
								   1, InputError)
		ta.assert_greater_or_equal(self.beam_ref_frame, 'beam ref. frame',
								   1, InputError)

		# Machine parameter assertion
		ta.assert_greater_or_equal(self.h_num, 'harmonic number',
								   1, MachineParameterError)
		ta.assert_greater_or_equal(self.h_ratio, 'harmonic ratio',
								   1, MachineParameterError)
		ta.assert_greater(self.b0, 'B field (B0)',
						  0, MachineParameterError)
		ta.assert_greater(self.mean_orbit_rad, "mean orbit radius",
						  0, MachineParameterError)
		ta.assert_greater(self.bending_rad, "Bending radius",
						  0, MachineParameterError)
		ta.assert_greater(self.e_rest, 'rest energy',
						  0, MachineParameterError)

		# Space charge parameter assertion
		ta.assert_greater_or_equal(self.pickup_sensitivity,
								   'pick-up sensitivity',
								   0, SpaceChargeParameterError)
		ta.assert_greater_or_equal(self.g_coupling, 'g_coupling',
								   0, SpaceChargeParameterError,
								   'NB: g_coupling:'
								   'geometrical coupling coefficient')

	# Asserting that some of the parameters calculated are valid
	def _assert_parameters(self):
		# Calculated parameters
		ta.assert_greater_or_equal(self.profile_length, 'profile length', 0,
								   InputError,
								   f'Make sure that the sum of post- and'
								   f'pre-skip length is less'
								   f'than the frame length\n'
								   f'frame length: {self.framelength}\n'
								   f'pre-skip length: {self.preskip_length}\n'
								   f'post-skip length: {self.postskip_length}')

		ta.assert_array_shape_equal([self.time_at_turn,
									 self.omega_rev0,
									 self.phi0,
									 self.dphase,
									 self.deltaE0,
									 self.beta0,
									 self.eta0,
									 self.e0],
									['time_at_turn',
									 'omega_re0',
									 'phi0',
									 'dphase',
									 'deltaE0',
									 'beta0',
									 'eta0',
									 'e0'],
									(self._calc_number_of_turns() + 1, ))

	# Calculating total number of machine turns
	def _calc_number_of_turns(self):
		all_turns = (self.framecount - self.frame_skipcount - 1) * self.dturns
		ta.assert_greater(all_turns, 'all_turns', 0, InputError,
						  'Make sure that frame skip-count'
						  'do not exceed number of frames')
		return all_turns
