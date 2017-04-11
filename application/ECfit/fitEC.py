""" Functions for fitting an equivalent circuit analog to data

Loosely based off of a matlab routine, Zfit, from Jean-Luc Dellis.

https://www.mathworks.com/matlabcentral/fileexchange/19460-zfit

"""

from application.ECfit.circuit_elements import *
from scipy.optimize import leastsq
import numpy as np
import sys


def equivalent_circuit(data, circuit_string, initial_guess):
    """ Main function for fitting an equivalent circuit to data

    Parameters
    -----------------
    data : list of tuples
        list of (frequency, real impedance, imaginary impedance)

    circuit_string : string
        string defining the equivalent circuit to be fit

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
    Need to do a better job of handling errors in fitting.
    Currently, an error of -1 is returned.

    """

    circuit_string = circuit_string.replace('_', '')

    f = np.array([f for f, r, i in data])
    zr = np.array([r for f, r, i in data])
    zi = np.array([i for f, r, i in data])

    zrzi = zr + 1j*zi

    p_values, covar, _, _, ier = leastsq(residuals, initial_guess,
                                         args=(zrzi, f, circuit_string),
                                         maxfev=100000, ftol=1E-13,
                                         full_output=True)

    p_error = []
    if ier in [1, 2, 3, 4] and covar is not None:
        s_sq = (residuals(p_values, zrzi, f, circuit_string)**2).sum()
        p_cov = covar * s_sq/(len(zrzi) - len(p_values))
        for i, __ in enumerate(covar):
            p_error.append(np.absolute(p_cov[i][i])**0.5)
    else:
        p_error = len(p_values)*[-1]

    fit_zrzi = computeCircuit(circuit_string, p_values.tolist(), f.tolist())

    fit = list(zip(f, np.real(fit_zrzi), np.imag(fit_zrzi)))

    return p_values, p_error, fit


def residuals(param, Z, f, circuit_string):
    """ Calculates the residuals between a given circuit_string/parameters
    (fit) and `Z`/`f` (data). Minimized by scipy.leastsq()

    Parameters
    ----------
    param : array of floats
        parameters for evaluating the circuit

    Z : array of complex numbers
        impedance data being fit

    f : array of floats
        frequencies to evaluate

    circuit_string : str
        string defining the circuit

    Returns
    -------
    residual : ndarray
        returns array of size 2*len(f) with both real and imaginary residuals
    """
    err = Z - computeCircuit(circuit_string, param.tolist(), f.tolist())
    z1d = np.zeros(Z.size*2, dtype=np.float64)
    z1d[0:z1d.size:2] = err.real
    z1d[1:z1d.size:2] = err.imag
    if valid(circuit_string, param):
        return z1d
    else:
        return 1e6*np.ones(Z.size*2, dtype=np.float64)


def valid(circuit_string, param):
    """ checks validity of parameters

    Parameters
    ----------
    circuit_string : string
        string defining the circuit

    param : list
        list of parameter values

    Returns
    -------
    valid : boolean

    Notes
    -----
    All parameters are considered valid if they are greater than zero --
    except for E2 (the exponent of CPE) which also must be less than one.

    """

    p_string = [p for p in circuit_string if p not in 'ps(),-/']

    for i, (a, b) in enumerate(zip(p_string[::2], p_string[1::2])):
        if str(a+b) == "E2":
            if param[i] <= 0 or param[i] >= 1:
                return False
        else:
            if param[i] <= 0:
                return False

    return True


def computeCircuit(circuit_string, parameters, frequencies):
    """ evaluates a circuit string for a given set of parameters and frequencies

    Parameters
    ----------
    circuit_string : string
    parameters : list of floats
    frequencies : list of floats

    Returns
    -------
    array of floats
    """
    circuit = buildCircuit(circuit_string, parameters, frequencies)
    results = eval(circuit)
    return results  # np.column_stack((frequencies, results))


def buildCircuit(circuit_string, parameters, frequencies):
    """ transforms a circuit_string, parameters, and frequencies into a string
    that can be evaluated

    Parameters
    ----------
    circuit_string : str
    parameters : list of floats
    frequencies : list of floats

    Returns
    -------
    eval_string : str
        Python expression for calculating the resulting fit
    """

    series_string = "s(["
    for elem in circuit_string.split("-"):
        element_string = ""
        if "p" in elem:
            parallel_string = "p(("
            for par in elem.strip("p()").split(","):
                param_string = ""
                elem_type = par[0]
                elem_number = len(par.split("/"))

                param_string += str(parameters[0:elem_number])
                parameters = parameters[elem_number:]

                new_elem = (elem_type + "(" + param_string + "," +
                                        str(frequencies) + "),")
                parallel_string += new_elem

            element_string = parallel_string.strip(",") + "))"
        else:
            param_string = ""
            elem_type = elem[0]
            elem_number = len(elem.split("/"))

            param_string += str(parameters[0:elem_number])
            parameters = parameters[elem_number:]

            element_string = (elem_type + "(" + param_string + "," +
                                          str(frequencies) + ")")

        series_string += element_string + ","

    eval_string = series_string.strip(",") + "])"

    return eval_string
