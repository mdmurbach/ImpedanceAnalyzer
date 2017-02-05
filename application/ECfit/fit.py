from __future__ import print_function
from utilities import parseData, computeCircuit, residuals
from scipy.optimize import leastsq
import numpy as np
import sys

def equivalent_circuit(data, circuit_string, initial_guess):
    """ Fits an equivalent circuit to data

    Parameters
    -----------------
    data : list of tuples
        list of (frequency, real impedance, imaginary impedance)

    circuit_string : string
        string defining the equivalent circuit to be fit. see crossref for details

    initial_guess : list of floats
        initial guesses for the fit parameters

    Returns
    ------------
    fit : list of tuples
        list of (frequency, real impedance, imaginary impedance)

    p_values : list of floats
        best fit parameters for specified equivalent circuit

    p_errors : list of floats
        error estimates for fit parameters

    Notes
    ---------


    """

    f, zr, zi = parseData(data)

    zrzi = zr + 1j*zi

    p_values, covar, info, errmsg, ier = leastsq(residuals, initial_guess, args=(zrzi, f, circuit_string), maxfev=100000, ftol=1E-13,  full_output=True)

    p_error = []
    if ier in [1,2,3,4]:
        s_sq = ((residuals(p_values, zrzi, f, circuit_string)**2).sum())/(len(zrzi) - len(p_values))
        p_cov = covar * s_sq
        for i, __ in enumerate(covar):
            try:
              p_error.append(np.absolute(p_cov[i][i])**0.5)
            except:
              p_error.append(0.0)
    else:
        p_error = len(p_values)*[-1]

    fit_data_1 = computeCircuit(circuit_string, p_values.tolist(), f.tolist())

    fit_zrzi = [a[1] for a in fit_data_1]

    fit = zip(f, np.real(fit_zrzi), np.imag(fit_zrzi))

    r_squared = 1

    return p_values, p_error, fit#, r_squared
