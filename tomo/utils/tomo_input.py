import numpy as np
import os
import sys

from .. import profiles as profs
from .. import machine as mach
from . import data_treatment as threat
from . import exceptions as expt
from . import assertions as asrt

# Some constants for the input file containing machine parameters
PARAMETER_LENGTH = 98
RAW_DATA_FILE_IDX = 12
OUTPUT_DIR_IDX = 14

# Class for storing raw data and information on 
# how to threat them, based on a Fortran style input file.
class Frames:

    def __init__(self, framecount, framelength, skip_frames, skip_bins_start,
                 skip_bins_end, rebin, dtbin, raw_data_path=''):
        self.raw_data_path = raw_data_path
        self.nframes = framecount
        self.nbins_frame = framelength
        self.skip_frames = skip_frames
        self.skip_bins_start = skip_bins_start
        self.skip_bins_end = skip_bins_end
        self.rebin = rebin
        self.sampling_time = dtbin

    @property
    def raw_data(self):
        if self._raw_data is not None:
            return np.copy(self._raw_data)
        else:
            return None
    
    @raw_data.setter
    def raw_data(self, in_raw_data):
        if not hasattr(in_raw_data, '__iter__'):
            raise expt.RawDataImportError('Raw data should be iterable')

        ndata = self.nframes * self.nbins_frame
        if len(in_raw_data) == ndata:
            self._raw_data = np.array(in_raw_data)
        else:
            expt.RawDataImportError(f'Raw data has length {len(raw_data)}.\n'
                                    f'expected length: {ndata}')

    def nprofs(self):
        return self.nframes - self.skip_frames

    def nbins(self):
        return self.nbins_frame - self.skip_bins_start - self.skip_bins_end

    # Convert from one-dimentional list of raw data to waterfall.
    # Works on a copy of the raw data
    def to_waterfall(self, raw_data):
        waterfall = self._assert_raw_data(raw_data)
        asrt.assert_frame_inputs(self)

        waterfall = waterfall.reshape((self.nframes, self.nbins_frame))
        waterfall = waterfall[self.skip_frames:]
        
        if self.skip_bins_end > 0:
            waterfall = waterfall[:, self.skip_bins_start:
                                    -self.skip_bins_end]
        else:
            waterfall = waterfall[:, self.skip_bins_start:]
        return waterfall

    def _assert_raw_data(self, raw_data):
        if not hasattr(raw_data, '__iter__'):
            raise expt.RawDataImportError('Raw data should be iterable')

        ndata = self.nframes * self.nbins_frame
        if not len(raw_data) == ndata:
            raise expt.RawDataImportError(
                    f'Raw data has length {len(raw_data)}.\n'
                    f'expected length: {ndata}')
        
        return np.array(raw_data)
            


# Function to be called from main.
# Lets the user give input using stdin or via args
def get_user_input():
    if len(sys.argv) > 1:
        read = _get_input_args()
    else:
        read = _get_input_stdin()
    return _split_input(read)


# Recieve path to input file via sys.argv.
# Can also recieve the path to the output directory.
def _get_input_args():
    input_file_pth = sys.argv[1]
    
    if not os.path.isfile(input_file_pth):
        raise expt.InputError(f'The input file: "{input_file_pth}" '
                              f'does not exist!')
    
    with open(input_file_pth, 'r') as f:
        read = f.readlines()
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        if os.path.isdir(output_dir):
            read[OUTPUT_DIR_IDX] = output_dir
        else:
            raise expt.InputError(f'The chosen output directory: '
                                  f'"{output_dir}" does not exist!')
    return np.array(read)


# Read machine parameters via stdin.
# Here the measured data must be pipelined in the same file as
# the machine parameters.
def _get_input_stdin():
    read = []
    finished = False
    piped_raw_data = False
    
    line_num = 0
    ndata_points = PARAMETER_LENGTH

    while line_num < ndata_points:
        read.append(sys.stdin.readline())
        if line_num == RAW_DATA_FILE_IDX:
            if 'pipe' in read[-1]:
                piped_raw_data = True
        if piped_raw_data:
            if line_num == 16:
                nframes = int(read[-1])
            if line_num == 20:
                nbins = int(read[-1])
                ndata_points += nframes * nbins
        if line_num == ndata_points:
            finished = True
        line_num += 1
    return np.array(read)


# Splits the read input data to machine parameters and raw data.
# If the raw data is not already read from the input file, the
#  data will be found in the file given by the parameter file.
def _split_input(read_input):
    nframes_idx = 16
    nbins_idx = 20
    ndata = 0
    read_parameters = None
    read_data = None
        
    try:
        read_parameters = read_input[:PARAMETER_LENGTH]
        ndata = (int(read_parameters[nbins_idx])
                 * int(read_parameters[nframes_idx]))
        for i in range(PARAMETER_LENGTH):
            read_parameters[i] = read_parameters[i].strip('\r\n')
    except:
        err_msg = 'Something went wrong while accessing machine parameters.'
        raise expt.InputError(err_msg)

    if read_parameters[RAW_DATA_FILE_IDX] == 'pipe':
        try:
            read_data = np.array(read_input[PARAMETER_LENGTH:], dtype=float)
        except:
            err_msg = 'Pipelined raw-data could not be casted to float.'
            raise expt.InputError(err_msg)
    else:
        try:
            read_data = np.genfromtxt(read_parameters[RAW_DATA_FILE_IDX],
                                      dtype=float)
        except FileNotFoundError:
            err_msg = f'The given file path for the raw-data:\n'\
                      f'{read_parameters[RAW_DATA_FILE_IDX]}\n'\
                      f'Could not be found'
            raise FileNotFoundError(err_msg)
        except Exception:
            err_msg = 'Something went wrong while loading raw_data.'
            raise Exception(err_msg)
            

    if not len(read_data) == ndata:
        raise expt.InputError(f'Wrong amount of datapoints loaded.\n'
                              f'Expected: {ndata}\n'
                              f'Loaded:   {len(read_data)}')


    return read_parameters, read_data


# Function to convert from array containing the lines in an input file
#  to a partially filled machine object.
# The array must contain a direct read from an input file.
# Some variables are subtracted by one.
#  This is in order to convert from Fortran to C style indexing.
#  Fortran counts (by defalut) from 1.
def txt_input_to_machine(input_array):
    if not hasattr(input_array , '__iter__'):
        raise expt.InputError('Input should be iterable') 
    if len(input_array) != PARAMETER_LENGTH:
        raise expt.InputError(f'Input array be of length {PARAMETER_LENGTH}, '
                              f'containin every line of input file '
                              f'(not including raw-data).')

    for i in range(len(input_array)):
            input_array[i] = input_array[i].strip('\r\n')


    fargs = {
             'raw_data_path':       input_array[12],
             'framecount':          int(input_array[16]),
             'skip_frames':         int(input_array[18]),
             'framelength':         int(input_array[20]),
             'dtbin':               float(input_array[22]),
             'skip_bins_start':     int(input_array[26]),
             'skip_bins_end':       int(input_array[28]),
             'rebin':               int(input_array[36])
            }

    frame = Frames(**fargs)
    nprofiles = frame.nprofs()
    nbins = frame.nbins()

    min_dt, max_dt = _min_max_dt(nbins, input_array)

    margs = {
             'output_dir':          input_array[14],
             'dtbin':               float(input_array[22]),
             'dturns':              int(input_array[24]),
             'xat0':                float(input_array[39]),
             'demax':               float(input_array[41]),
             'filmstart':           int(input_array[43]) - 1,
             'filmstop':            int(input_array[45]),
             'filmstep':            int(input_array[47]),
             'niter':               int(input_array[49]),
             'snpt':                int(input_array[51]),
             'full_pp_flag':        bool(int(input_array[53])),
             'beam_ref_frame':      int(input_array[55]) - 1,
             'machine_ref_frame':   int(input_array[57]) - 1,
             'vrf1':                float(input_array[61]),
             'vrf1dot':             float(input_array[63]),
             'vrf2':                float(input_array[65]),
             'vrf2dot':             float(input_array[67]),
             'h_num':               float(input_array[69]),
             'h_ratio':             float(input_array[71]),
             'phi12':               float(input_array[73]),
             'b0':                  float(input_array[75]),
             'bdot':                float(input_array[77]),
             'mean_orbit_radius':   float(input_array[79]),
             'bending_radius':      float(input_array[81]),
             'transitional_gamma':  float(input_array[83]),
             'rest_energy':         float(input_array[85]),
             'charge':              float(input_array[87]),
             'self_field_flag':     bool(int(input_array[91])),
             'g_coupling':          float(input_array[93]),
             'zwall_over_n':        float(input_array[95]),
             'pickup_sensitivity':  float(input_array[97]),
             'nprofiles':           nprofiles,
             'nbins':               nbins,
             'min_dt':              min_dt,
             'max_dt':              max_dt
            }

    machine = mach.Machine(**margs)
    
    return machine, frame

def _min_max_dt(nbins, input_array):
    dtbin = float(input_array[22])
    min_dt_bin = int(input_array[31])
    max_dt_bin = int(input_array[34])

    min_dt = min_dt_bin * dtbin
    max_dt = (nbins - max_dt_bin) * dtbin
    return min_dt, max_dt


def raw_data_to_profiles(waterfall, machine, rbn, sampling_time):
    if not hasattr(waterfall, '__iter__'):
        raise expt.WaterfallError('Waterfall should be an iterable')
    waterfall = np.array(waterfall)
    waterfall[:] -= threat.calc_baseline_ftn(
                        waterfall, machine.beam_ref_frame)
    waterfall = threat.rebin(waterfall, rbn, machine)
    return profs.Profiles(machine, sampling_time, waterfall)
    
    