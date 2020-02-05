"""Module containing fundamental physics formulas.

:Author(s): **Christoffer Hjertø Grindheim**
"""

import numpy as np
from scipy import optimize, constants


def b_to_e(machine):
    '''
    Calculates the energy for a particle
    in a circular machine at dipole field B.
    
    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.
    
    Returns
    -------
    energy: float
        Energy calculated from B-field.
    '''
    return np.sqrt((machine.q
                    * machine.b0
                    * machine.bending_rad
                    * constants.c)**2
                   + machine.e_rest**2)


def lorenz_beta(machine, rf_turn):
    '''Calculates Lorenz beta factor (v/c) at a turn.

    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.
    rf_turn: int
        At whch RF turn the calculation should happen.

    Returns
    -------
    Lorentz beta: float
        The Lorenz beta factor (v/c) at a given turn.
    '''
    return np.sqrt(1.0 - (float(machine.e_rest) /
                          machine.e0[rf_turn])**2)


def rfvolt_rf1(phi, machine, rf_turn):
    '''Calculates RF voltage1 seen by paricle.
    Needed by the Newton root finder to calculate phi0.

    Parameters
    ----------
    phi: float
        Phase where voltage is to be calculated.
    machine: Machine
        Stores machine parameters and reconstruction settings.
    rf_turn: int
        At whch RF turn the calculation should happen.

    Returns
    -------
    voltage: float
        Voltage from RF system 1, as seen by particle.
    '''
    turn_time = machine.time_at_turn[rf_turn]
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)
    q_sign = np.sign([machine.q])
    return (v1 * np.sin(phi)
            - 2 * np.pi * machine.mean_orbit_rad
                * machine.bending_rad * machine.bdot * q_sign[0])


def drfvolt_rf1(phi, machine, rf_turn):
    '''Calculates derivative of RF voltage1 seen by paricle.
    Needed by the Newton root finder to calculate phi0.

    Parameters
    ----------
    phi: float
        Phase where voltage is to be calculated.
    machine: Machine
        Stores machine parameters and reconstruction settings.
    rf_turn: int
        At whch RF turn the calculation should happen.

    Returns
    -------
    Derivative of voltage: float
        Time-derivative of voltage from RF system 1, as seen by particle.
    '''
    turn_time = machine.time_at_turn[rf_turn]
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)
    return v1 * np.cos(phi)


def rf_voltage(phi, machine, rf_turn):
    '''Calculates RF voltage from both RF systems seen by paricle.
    Needed by the Newton root finder to calculate phi0.

    Parameters
    ----------
    phi: float
        Phase where voltage is to be calculated.
    machine: Machine
        Stores machine parameters and reconstruction settings.
    rf_turn: int
        At whch RF turn the calculation should happen.

    Returns
    -------
    voltage: float
        Voltage from both RF systems, as seen by particle.
    '''
    turn_time = machine.time_at_turn[rf_turn]
    q_sign = np.sign([machine.q])
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)
    v2 = vrft(machine.vrf2, machine.vrf2dot, turn_time)
    return (v1 * np.sin(phi)
            + (v2 * np.sin(machine.h_ratio * (phi - machine.phi12)))
            - (np.pi * 2 * machine.mean_orbit_rad
               * machine.bending_rad * machine.bdot * q_sign[0]))


def drf_voltage(phi, machine, rf_turn):
    '''Calculates derivative of RF voltage from both RF systems seen by paricle.
    Needed by the Newton root finder to calculate phi0.

    Parameters
    ----------
    phi: float
        Phase where voltage is to be calculated.
    machine: Machine
        Stores machine parameters and reconstruction settings.
    rf_turn: int
        At whch RF turn the calculation should happen.

    Returns
    -------
    Derivative of voltage: float
        Derivative of voltage from both RF systems, as seen by particle.
    '''
    turn_time = machine.time_at_turn[rf_turn]
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)
    v2 = vrft(machine.vrf2, machine.vrf2dot, turn_time)
    return (v1 * np.cos(phi)
            + machine.h_ratio
            * v2
            * np.cos(machine.h_ratio
                     * (phi - machine.phi12)))


def short_rf_voltage_formula(phi, vrf1, vrf1dot, vrf2, vrf2dot,
                             h_ratio, phi12, time_at_turn, rf_turn):
    '''RF voltage formula without calculating the difference in E0
    '''
    turn_time = time_at_turn[rf_turn]
    v1 = vrft(vrf1, vrf1dot, turn_time)
    v2 = vrft(vrf2, vrf2dot, turn_time)
    temp = v1 * np.sin(phi)
    temp += v2 * np.sin(h_ratio * (phi - phi12))
    return temp


def vrft(vrf, vrf_dot, turn_time):
    '''Calculates the RF peak voltage at turn rf_turn
    assuming a linear voltage function.
    Time=0 at machine reference frame.

    Parameters
    ----------
    vrf: float
        Peak voltage.
    vrf_dot: float
        Time derivative of RF voltage.
    time_at_turn: float
        time at turn for which the RF voltage should be calculated.
    
    Returns
    -------
    RF voltage: float
        RF voltage at turn.
    '''
    return vrf + vrf_dot * turn_time


def find_synch_phase(machine, rf_turn, phi_lower, phi_upper):
    '''Uses the Newton-Raphson root finder to estimate
    the synchronous pase for a particle on the normal orbit.
    
    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.
    rf_turn: int
        At whch RF turn the calculation should happen.
    phi_lower:
        Lower phase boundary
    phi_upper:
        Upper phase boundary

    Returns
    -------
    synchronous phase: float
        Synchronous phase given in radients.
    '''
    phi_start = optimize.newton(func=rfvolt_rf1,
                                x0=(phi_lower + phi_upper) / 2.0,
                                fprime=drfvolt_rf1,
                                tol=0.0001,
                                maxiter=100,
                                args=(machine, rf_turn))
    synch_phase = optimize.newton(func=rf_voltage,
                                  x0=phi_start,
                                  fprime=drf_voltage,
                                  tol=0.0001,
                                  maxiter=100,
                                  args=(machine, rf_turn))
    return synch_phase


def find_phi_lower_upper(machine, rf_turn):
    '''Calculates lower and upper phase of RF voltage 
    for use in estimation of phi0.

    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.
    rf_turn: int
        At whch RF turn the calculation should happen.

    Returns
    -------
    phi lower: float
        Lower phase boundary [rad].
    phi upper: float
        Upper phase boundary [rad].

    '''
    condition = (machine.q
                 * (machine.trans_gamma
                    - (machine.e0[rf_turn]
                        / float(machine.e_rest)))) > 0
    if condition:
        phi_lower = -np.pi
        phi_upper = np.pi
    else:
        phi_lower = 0.0
        phi_upper = 2 * np.pi
    return phi_lower, phi_upper


def phase_slip_factor(machine):
    '''Calculates phase slip factor at each turn.

    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.

    Returns
    -------
    phase slip factor: ndarray, float
    '''
    return (1.0 - machine.beta0**2) - machine.trans_gamma**(-2)


def find_dphase(machine):
    '''Calculates coefficient needed for drift calculation during tracking.
    The drift coefficient is calculated for each machine turn.
    
    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.

    Returns
    -------
    phase slip factor: ndarray, float
        Phase slip factor for each machine turn.
    '''
    return (2 * np.pi * machine.h_num * machine.eta0
            / (machine.e0 * machine.beta0**2))


def revolution_freq(machine):
    '''Calculate revolution frequency for each turn.

    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.
    
    Returns
    -------
    revolution frequency: ndarray, float
        Revolution frequency at each machine turn.
    '''
    return machine.beta0 * constants.c / machine.mean_orbit_rad


def calc_self_field_coeffs(machine):
    '''Calculates self-field coefficient for each profile.
    Needed for calculation of self-fields.
    
    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.

    Returns
    -------
    Self-field coefficients: ndarray, float
        Self-field coefficient at each time frame.
    '''
    sfc = np.zeros(machine.nprofiles)
    for i in range(machine.nprofiles):
        this_turn = i * machine.dturns
        sfc[i] = ((constants.e / machine.omega_rev0[this_turn])
                  * ((1.0 / machine.beta0[this_turn]
                     - machine.beta0[this_turn])
                     * machine.g_coupling * np.pi * 2.0e-7 * constants.c
                     - machine.zwall_over_n)
                  / machine.dtbin**2)
    return sfc


def phase_low(phase, machine, bunch_phaselength, rf_turn):
    '''Calculates potential energy at phase.
    Needed for estimation of x coordinate of synchronous particle.

    Write about term1, term2, term3.

    Parameters
    ----------
    phase: float
        Phase where potential energy is calculated.
    machine: Machine
        Stores machine parameters and reconstruction settings.
    bunch_phaselength: float
        Length of bunch [rad]
    rf_turn: int
        At whch RF turn the calculation should happen.
    '''
    term1 = (machine.vrf2
             * (np.cos(machine.h_ratio
                       * (phase
                          + bunch_phaselength
                          - machine.phi12))
                - np.cos(machine.h_ratio
                         * (phase - machine.phi12)))
             / machine.h_ratio)
    term2 = (machine.vrf1
             * (np.cos(phase + bunch_phaselength)
                - np.cos(phase)))
    term3 = (bunch_phaselength
             * short_rf_voltage_formula(
                machine.phi0[rf_turn], machine.vrf1, machine.vrf1dot,
                machine.vrf2, machine.vrf2dot, machine.h_ratio,
                machine.phi12, machine.time_at_turn, rf_turn))
    return term1 + term2 + term3


def dphase_low(phase, machine, bunch_phaselength, *args):
    '''Calculates derivative of the phase_low function.
    The function is needed for estimation of x coordinate 
    of synchronous particle.

    Parameters
    ----------
    phase: float
        Phase where potential energy is calculated.
    machine: Machine
        Stores machine parameters and reconstruction settings.
    bunch_phaselength: float
        Length of bunch [rad]
    args: None
        Not used, needed for implementation of Newton-Raphson root finder.
    '''
    return (-1.0 * machine.vrf2
            * (np.sin(machine.h_ratio
                      * (phase + bunch_phaselength - machine.phi12))
                - np.sin(machine.h_ratio * (phase - machine.phi12)))
            - machine.vrf1
            * (np.sin(phase + bunch_phaselength) - np.sin(phase)))
