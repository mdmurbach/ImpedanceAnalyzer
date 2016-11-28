from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import cmath
from scipy.optimize import leastsq, basinhopping, brute, minimize

def fitP2D(data):
    freq = np.array([run[0] for run in data])
    real = np.array([run[1] for run in data])
    imag = np.array([run[2] for run in data])

    mag =  np.array([np.sqrt(run[1]**2 + run[2]**2) for run in data])
    phase =  np.array([np.arctan2(-run[2], run[1]) for run in data])


    Z = pd.read_pickle('./14067-Z.pkl')

    min_f = min(freq)
    max_f = max(freq)

    frequencies = [f for f in Z.columns if min_f <= f <= max_f]

    to_fit = pd.DataFrame(index=frequencies, columns=['Zmag', 'Zph'])

    for frequency in frequencies:
        idx = np.argmin(np.abs(frequency - freq))

        m_mag = (mag[idx+1] - mag[idx-1])/(freq[idx+1] - freq[idx-1])
        b_mag = mag[idx+1] - m_mag*freq[idx+1]

        y_mag = m_mag*frequency + b_mag

        m_ph = (phase[idx+1] - phase[idx-1])/(freq[idx+1] - freq[idx-1])
        b_ph = phase[idx+1] - m_ph*freq[idx+1]

        y_ph = m_ph*frequency + b_ph

        to_fit.loc[frequency, 'Zmag'] = y_mag
        to_fit.loc[frequency, 'Zph'] = y_ph

    to_fit['Zreal'] = to_fit.Zmag*(to_fit.Zph.map(np.cos))
    to_fit['Zimag'] = to_fit.Zmag*(to_fit.Zph.map(np.sin))

    def residual(scale, Z11_model, Z11_exp):
        '''
        Returns average distance of error between the model and experimental data
        '''
        return (1./len(Z11_model))*np.sqrt(sum((Z11_exp.map(np.real) - scale*np.real(Z11_model))**2 + (Z11_exp.map(np.imag) - scale*np.imag(Z11_model))**2))

    def fit_model(Z11_model):
        res = minimize(residual, 10.0, args=(Z11_model, Z11_exp), tol=1e-5)
        return [res.x, res.fun]

    Z11_exp = to_fit.Zreal + 1j*to_fit.Zimag

    print(Z11_exp, file=sys.stderr)
    print(Z11_exp, file=sys.stderr)

    results = Z.loc[:, frequencies].apply(fit_model, axis=1)

    def split_scale(input):
        return input[0][0]

    def split_mse(input):
        return input[1]

    res_df = pd.DataFrame(index = results.index, columns=['scale', 'mse'])

    res_df['scale'] = results.map(split_scale)
    res_df['mse'] = results.map(split_mse)

    sorted_res_df = res_df.sort_values(['mse'])

    best_fit = sorted_res_df.index[0]
    Z11_model = sorted_res_df['scale'].loc[best_fit]*Z.loc[best_fit]

    print(Z11_model, file=sys.stderr)

    fit = zip(Z11_model.index, Z11_model.map(np.real).tolist(), (-1*Z11_model.map(np.imag)).tolist())

    print(fit, file=sys.stderr)

    return fit

def fitEquivalentCircuit(data, p0):
    print("fitting circuit")

    # Define Circuit and Initial Parameters
    global circuit_string

    circuit_string ='s(R1,p(s(R1,W2),E2))'

    # Define circuits to compare
    # circuit_string = randles_circuit
    # parameters = p0

    f = np.array([run[0] for run in data])
    real = np.array([run[1] for run in data])
    imag = np.array([run[2] for run in data])

    # Calculate best guesses
    r1 = real[np.argmax(f)]
    r2 = real.mean() - r1

    c1 = 1/(f[f>1][np.argmax(np.abs(np.arctan2(imag[f>1],real[f>1])))]*r1)

    parameters = [r1, r2, p0[2], p0[3], c1, p0[5]]

    print(parameters, file=sys.stderr)

    freq = np.array([a for a,b,c in data])
    zr = np.array([b for a,b,c in data])
    zi =np.array([c for a,b,c in data])
    zrzi = zr + 1j*zi

    # Simulates Initial Conditions and Performs Least
    # Squares fit of circuit(s)
    sim_data = compute_circuit(parameters, circuit_string, freq)

    plsq, covar, info, errmsg, ier = leastsq(residuals, parameters, args=(zrzi, freq), maxfev=100000,
                                            ftol=1E-13,  full_output=True)

    s_sq = ((residuals(plsq, zrzi, freq)**2).sum())/(len(zrzi) - len(plsq))
    p_cov = covar * s_sq

    error = []
    for i in range(len(covar)):
        try:
          error.append(np.absolute(p_cov[i][i])**0.5)
        except:
          error.append( 0.00 )

    fit_data_1 = compute_circuit(plsq.tolist(), circuit_string, freq)

    fit_zrzi = [a[1] for a in fit_data_1]

    fit = zip(freq.tolist(), np.real(fit_zrzi).tolist(), np.imag(fit_zrzi).tolist())

    return plsq.tolist(), error, fit

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
