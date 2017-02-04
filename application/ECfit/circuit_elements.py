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

    .. math::

        Z = \\frac{1}{C \\times j 2 \\pi f}

     """
    f = np.array(f)
    return 1.0/(p[0]*1j*2*np.pi*f)


def W(p, f):
    """ defines a Finite-length Warburg Element

    Notes
    ---------
    .. math::
        Z = \\frac{R}{\\sqrt{ T \\times j 2 \\pi f}} \\coth{\\sqrt{T \\times j 2 \\pi f }}

    where :math:`R` = p[0] (Ohms) and :math:`T` = p[1] (sec) = :math:`\\frac{L^2}{D}`

    """
    f = np.array(f)
    fx = np.vectorize(lambda y: p[0]/(np.sqrt(p[1]*1j*2*np.pi*y)*cmath.tanh(np.sqrt(p[1]*1j*2*np.pi*y))))
    z = fx(f)
    return z


def E(p, f):
    """ defines a constant phase element

    Notes
    ---------
    .. math::

        Z = \\frac{1}{C \\times (j 2 \\pi f)^n}

    where :math:`C` = p[0] is the capacitance and :math:`n` = p[1] is the exponential factor

    """
    return np.array([1.0/(p[0]*(1j*2*np.pi*w)**p[1]) for w in f])


def G(p, f):
    """ defines a Gerischer Element

    Notes
    ---------
    .. math::

        Z = \\frac{1}{Y \\times \\sqrt{K + j 2 \\pi f }}

     """
    return np.array([1.0/(p[0]*np.sqrt(p[1] + 1j*2*np.pi*w)) for w in f])
