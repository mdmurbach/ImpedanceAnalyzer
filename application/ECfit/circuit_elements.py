from __future__ import print_function
import numpy as np
import cmath
import sys

def s(series):
    """ sums elements in series

    Notes
    ---------
    .. math::
        Z = Z_1 + Z_2 + ... + Z_n

    """
    z = len(series[0])*[0 + 0*1j]
    for elem in series:
        z += elem
    return z

def p(parallel):
    """ adds elements in parallel

    Notes
    ---------
    .. math::

        Z = \\frac{1}{\\frac{1}{Z_1} + \\frac{1}{Z_2} + ... + \\frac{1}{Z_n}}

     """
    z = len(parallel[0])*[0 + 0*1j]
    for elem in parallel:
        z += 1/elem
    return 1/z


# Resistor
def R(p, f):
    """ defines a resistor

    Notes
    ---------
    .. math::

        Z = R

    """
    return np.array(len(f)*[p[0]])


# Capacitor
def C(p, f):
    """ defines a capacitor

    .. math::

        Z = \\frac{1}{C \\times j 2 \\pi f}

     """

    omega = 2*np.pi*np.array(f)
    C = p[0]

    return 1.0/(C*1j*omega)


def W(p, f):
    """ defines a blocked boundary Finite-length Warburg Element

    Notes
    ---------
    .. math::
        Z = \\frac{R}{\\sqrt{ T \\times j 2 \\pi f}} \\coth{\\sqrt{T \\times j 2 \\pi f }}

    where :math:`R` = p[0] (Ohms) and :math:`T` = p[1] (sec) = :math:`\\frac{L^2}{D}`

    """

    omega = 2*np.pi*np.array(f)



    Zw = np.vectorize(lambda y: p[0]/(np.sqrt(p[1]*1j*y)*cmath.tanh(np.sqrt(p[1]*1j*y))))

    return Zw(omega)

def A(p, f):
    """ defines a semi-infinite Warburg element

    """

    omega = 2*np.pi*np.array(f)
    Aw = p[0]

    Zw = Aw*(1-1j)*np.sqrt(omega)

    return Zw

def E(p, f):
    """ defines a constant phase element

    Notes
    -----
    .. math::

        Z = \\frac{1}{Q \\times (j \\omega)^\\alpha}

    where :math:`Q` = p[0] and :math:`\\alpha` = p[1]. [1]_

    References
    ----------
    .. [1] Equation (13.1) from Orazem, M. E. & Tribollet, B. Electrochemical impedance spectroscopy. (Wiley, 2008).

    """

    # Parameters
    # ----------
    # p : list
    #     parameters for the circuit element
    # f : list
    #     frequencies for calculating the element impedance
    #
    # Returns
    # -------
    # Z : array
    #     impedance of the circuit element with the given parameters and frequency

    omega = 2*np.pi*np.array(f)
    Q = p[0]
    alpha = p[1]

    return 1.0/(Q*(1j*omega)**alpha)


def G(p, f):
    """ defines a Gerischer Element

    Notes
    ---------
    .. math::

        Z = \\frac{1}{Y \\times \\sqrt{K + j 2 \\pi f }}

     """

    omega = 2*np.pi*np.array(f)
    Z0 = p[0]
    k = p[1]

    return Z0/np.sqrt(k + 1j*omega)
