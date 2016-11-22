from __future__ import print_function
import sys
import numpy as np
import cmath
from scipy.optimize import leastsq, basinhopping, brute


def fitEquivalentCircuit(data, p0):
    print("fitting circuit")

    # Define Circuit and Initial Parameters
    global circuit_string

    randles_circuit ='s(R1,p(s(R1,W2),E2))'
    # randles_circuit_results = [ 'R_ct_1', 'Warburg_Z0', 'C_dl']
    # randles_circuit_parameters = [.1,   .35,   .01, 50,   .001,  .99]
    randles_circuit_parameters = p0 #[.01,   .01,   .01, 10,   .01,  .8]

    print('data: ' + str(data), file=sys.stderr)
    print('p0: ' + str(randles_circuit_parameters), file=sys.stderr)

    # Define circuits to compare
    circuit_string = randles_circuit
    # results_string = randles_circuit_results
    parameters = randles_circuit_parameters

    freq = np.array([a for a,b,c in data])
    zr = np.array([b for a,b,c in data])
    zi =np.array([c for a,b,c in data])
    zrzi = zr + 1j*zi


    # Simulates Initial Conditions and Performs Least
    # Squares fit of circuit(s)

    sim_data = compute_circuit(parameters, circuit_string, freq)

    plsq, covar, info, errmsg, ier = leastsq(residuals, parameters, args=(zrzi, freq), maxfev=100000,
                                            ftol=1E-13,  full_output=True)

    # ranges = [(.001, 1), (.001,1), (.001, 10),(.001, 1000),(.00001,.01),(.1,.99)]

    # x0, fval, grid, Jout = brute(residuals, ranges, args=(zrzi, freq), Ns=10, full_output=True, finish=None, disp=True)

    # fit_data_1 = compute_circuit(x0.tolist(), circuit_string, freq)

    fit_data_1 = compute_circuit(plsq.tolist(), circuit_string, freq)

    fit_zrzi = [a[1] for a in fit_data_1]

    # print(x0, file=sys.stderr)

    fit = zip(freq.tolist(), np.real(fit_zrzi).tolist(), np.imag(fit_zrzi).tolist())

    return plsq.tolist(), fit

# def residuals(param, y, x):
#     err = y - compute_circuit(param.tolist(), circuit_string, x)[:, 1]
#     z1d = np.zeros(y.size*2, dtype=np.float64)
#     z1d[0:z1d.size:2] = err.real
#     z1d[1:z1d.size:2] = err.imag
#     if valid(param):
#         return z1d.sum()
#     else:
#         return 1e6

def residuals(param, y, x):
    err = y - compute_circuit(param.tolist(), circuit_string, x)[:, 1]
    z1d = np.zeros(y.size*2, dtype=np.float64)
    z1d[0:z1d.size:2] = err.real
    z1d[1:z1d.size:2] = err.imag
    if valid(param):
        return z1d
    else:
        return 1e6*np.ones(y.size*2, dtype=np.float64)


def valid(param):
    if param[0] > 0 and param[1] > 0 and param[2] > 0 and param[3] > 0 and param[4] > 0 and param[5] > 0 and param[5] < 1:
        return True
    else:
        return False


# Load impedance data
def load_impedance(filename):
    data = np.loadtxt(filename)
    freq = data[:, 0]
    zrzi = data[:, 1]-1j*data[:, 2]
    return np.column_stack((freq, zrzi))


# ComputeCircuit
def compute_circuit(param, circuit, freq):
    a = ''.join(i for i in circuit if i not in 'ps(),')
    k = 0
    z = []
    for i in range(0, len(a), 2):
        nlp = int(a[i+1])
        localparam = param[0:nlp]
        param = param[nlp:]
        func = a[i] + '(' + str(localparam) + ',' + str(freq.tolist()) + ')'
        z.append(eval(func))
        circuit = circuit.replace(a[i]+a[i+1], 'z[' + str(k) + ']', 1)
        k += 1
    z = eval(circuit)
    return np.column_stack((freq, z))


# Resistor
def R(p, f):
    return np.array(len(f)*[p[0]])


# Capacitor
def C(p, f):
    f = np.array(f)
    return 1.0/(p[0]*1j*2*np.pi*f)


# Constant Phase Element
# p[0] = CPE_prefactor
# p[1] = CPE_exponent
def E(p, f):
    return np.array([1.0/(p[0]*(1j*2*np.pi*w)**p[1]) for w in f])

# Gerischer Element
# p[0] = Warburg Impedance
# p[1] = Time Constant
def G(p, f):
    return np.array([1.0/(p[0]*np.sqrt(p[1] + 1j*2*np.pi*w)) for w in f])


# Warburg impedance %% NOTE - np.tanh does not work with large numbers (must use cmath.tanh)
# p[0] = -dUdc*l_pos/(F*Deff)
# p[1] = tau_d = l_pos^2/Deff ~ 100
def W(p, f):
    f = np.array(f)
    # fx = np.vectorize(lambda y: p[0]/(np.sqrt(p[1]*1j*2*np.pi*y)*cmath.tanh(np.sqrt(p[1]*1j*2*np.pi*y))))
    # fx = np.vectorize(lambda y: cmath.tanh(p[1]*np.sqrt(1j*2*np.pi*y))/(p[0]*np.sqrt(1j*2*np.pi*y))) # Finite Warburg
    fx = np.vectorize(lambda y: p[0]*cmath.tanh(np.sqrt(p[1]*1j*2*np.pi*y))/(np.sqrt(p[1]*1j*2*np.pi*y) - cmath.tanh(np.sqrt(p[1]*1j*2*np.pi*y))))
    # fx = np.vectorize(lambda y: p[0]*(1-1j)/np.sqrt(y))
    z = fx(f)
    return z

# Standard Warburg (45 deg)
def Q(p, f):
    return np.array((1-1j)/(p[0]*np.sqrt(f)))


# Elements in parallel
def p(z1, z2):
    return [1.0/((1.0/z1[i])+(1.0/z2[i])) for i in range(len(z1))]


# Elements in series
def s(z1, z2):
    return [z1[i] + z2[i] for i in range(len(z1))]
