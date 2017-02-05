import numpy as np
import cmath

def s(series):
    """ sums elements in series

    Parameters
    ----------

    series : list of circuit elements to be summed

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

    Parameters
    ----------

    p : list of floats
        parameter values

    f : list of floats
        frequencies

    Returns
    -------

    Z : array of complex floats
        impedance of a capacitor

    .. math::

        Z = \\frac{1}{C \\times j 2 \\pi f}

     """

    omega = 2*np.pi*np.array(f)
    C = p[0]

    return 1.0/(C*1j*omega)


def W(p, f):
    """ defines a Finite-length Warburg Element

    Notes
    ---------
    .. math::
        Z = \\frac{R}{\\sqrt{ T \\times j 2 \\pi f}} \\coth{\\sqrt{T \\times j 2 \\pi f }}

    where :math:`R` = p[0] (Ohms) and :math:`T` = p[1] (sec) = :math:`\\frac{L^2}{D}`

    """

    omega = 2*np.pi*np.array(f)

    Warburg = np.vectorize(lambda y: p[0]/(np.sqrt(p[1]*1j*y)*cmath.tanh(np.sqrt(p[1]*1j*y))))

    return Warburg(omega)


def E(p, f):
    """ defines a constant phase element

    Notes
    ---------
    .. math::

        Z = \\frac{1}{C \\times (j 2 \\pi f)^n}

    where :math:`C` = p[0] is the capacitance and :math:`n` = p[1] is the exponential factor

    """

    omega = 2*np.pi*np.array(f)
    Q = p[0]
    n = p[1]

    return 1.0/(Q*(1j*omega)**n)


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
