import numpy as np
import cmath

def residuals(param, y, x, circuit_string):
    """ calculates the residuals for circuit fitting """
    err = y - computeCircuit(circuit_string, param.tolist(), x.tolist())[:, 1]
    z1d = np.zeros(y.size*2, dtype=np.float64)
    z1d[0:z1d.size:2] = err.real
    z1d[1:z1d.size:2] = err.imag
    if valid(circuit_string, param):
        # print(z1d.sum(), file=sys.stderr)
        return z1d
    else:
        return 1e6*np.ones(y.size*2, dtype=np.float64)


def valid(circuit_string, param):
    """ checks to see if parameters are all > 0 """

    p_string = [p for p in circuit_string if p not in 'ps(),-/']

    for i, (a, b) in enumerate(zip(p_string[::2], p_string[1::2])):
        if str(a+b) == "E2":
            if param[i] <= 0 or param[i] >= 1:
                return False
        else:
            if param[i] <= 0:
                return False

    return True
    # if all(param > 0):
    #
    # # if param[0] > 0 and param[1] > 0 and param[2] > 0 and param[3] > 0 and param[4] > 0 and param[5] > 0 and param[5] < 1:
    #     return True
    # else:
    #     return False


def computeCircuit(circuit_string, parameters, f):
    """ computes a circuit using eval """
    circuit = buildCircuit(circuit_string, parameters, f)
    results = eval(circuit)
    return np.column_stack((f, results))

def buildCircuit(circuit_string, parameters, f):
    """ builds a circuit to be evaluated with eval

    """
    series_string = "s(("
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

                parallel_string += elem_type + "(" + param_string + "," + str(f) + "),"

            element_string = parallel_string.strip(",") + "))"
        else:
            param_string = ""
            elem_type = elem[0]
            elem_number = len(elem.split("/"))

            param_string += str(parameters[0:elem_number])
            parameters = parameters[elem_number:]

            element_string = elem_type + "(" + param_string + "," + str(f) + ")"

        series_string += element_string + ","

    return series_string.strip(",") + "))"

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
