import numpy as np
from application.ECfit.circuit_elements import *

def parseData(data):
    """

    Parameters
    ----------

    data : list of tuples


    Returns
    -------

    f : np.array
        frequency

    zr : np.array
        real impedance

    zi : np.array
        imaginary impedance

    """
    f = np.array([a for a,b,c in data])
    zr = np.array([b for a,b,c in data])
    zi = np.array([c for a,b,c in data])

    return f, zr, zi

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


def computeCircuit(circuit_string, parameters, f):
    """ computes a circuit using eval """
    circuit = buildCircuit(circuit_string, parameters, f)
    results = eval(circuit)
    return np.column_stack((f, results))

def buildCircuit(circuit_string, parameters, f):
    """ builds a circuit to be evaluated with eval

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

    return series_string.strip(",") + "])"
