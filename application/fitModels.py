""" Provides functions for fitting equivalent circuit and physics-based models

"""

from __future__ import print_function
import sys
import cmath
import numpy as np
import pandas as pd
from scipy.optimize import leastsq, minimize
from scipy.interpolate import interp1d

def fitP2D(data):
    """ Fit physics-based model

    Parameters
    -----------------
    data : list of tuples
        list of tuples containing (frequency, real impedance, imaginary impedance)

    Returns
    ------------
    fit : list of tuples
        list of tuples (frequency, real impedance, imaginary impedance) containing the best fit

    sorted_results : pandas DataFrame
        sorted DataFrame with columns of ['run', 'scale', 'residual']

    Notes
    ---------


    """

    # take incoming list of tuples and create a dataframe
    # sorted by descending frequency
    exp_data = pd.DataFrame(data, columns=['f', 'real', 'imag'])
    exp_data['mag'] = exp_data.apply(lambda x: np.sqrt(x[1]**2 + x[2]**2), axis=1)
    exp_data['phase'] = exp_data.apply(lambda x: np.arctan2(x[2], x[1]), axis=1)
    exp_data.sort_values(by='f', ascending=False, inplace=True)
    exp_data.index = range(len(exp_data))

    # read in all of the simulation results
    Z = pd.read_pickle('application/static/data/17190-Z.pkl')

    # find the frequencies that fall within the experimental data and create
    # the DataFrame, to_fit, to store interpolated experimental data for fitting
    min_f, max_f = min(exp_data['f']), max(exp_data['f'])
    freq_mask = sorted([f for f in Z.columns if min_f <= f <= max_f], reverse=True)

    to_fit = pd.DataFrame(index=freq_mask, columns=['mag', 'ph'])

    # if the frequency isn't already within 1% of the simulation frequencies,
    # quadratically interpolate the nearest four points in the magnitude and phase
    for frequency in to_fit.index:
        exact = exp_data[exp_data['f'].between(.99*frequency, 1.01*frequency)]
        if not exact.empty:
            to_fit.loc[frequency, 'mag'] = np.asscalar(exact['mag'])
            to_fit.loc[frequency, 'ph'] = np.asscalar(exact['phase'])
        else:
            idx = np.argmin(np.abs(frequency - exp_data['f']))

            x = exp_data['f'].iloc[idx-2:idx+3]
            y_mag = exp_data['mag'].iloc[idx-2:idx+3]
            y_phase = exp_data['phase'].iloc[idx-2:idx+3]

            mag = interp1d(x, y_mag, kind='quadratic')
            phase = interp1d(x, y_phase, kind='quadratic')

            to_fit.loc[frequency, 'mag'] = mag(frequency)
            to_fit.loc[frequency, 'ph'] = phase(frequency)

    to_fit['real'] = to_fit.mag*(to_fit.ph.map(np.cos))
    to_fit['imag'] = to_fit.mag*(to_fit.ph.map(np.sin))

    # if the imaginary impedance crosses zeros, finds the crossover
    # and includes it in to_fit as the model's 10**5 Hz entry
    crossover = exp_data[exp_data['imag'] > 0]

    if crossover.index.tolist():
        index = crossover.index.tolist()[-1]

        x = exp_data['imag'].loc[index-2:index+3]
        y = exp_data['real'].loc[index-2:index+3]

        hf = interp1d(x,y, kind='quadratic')

        Zreal_hf = np.asscalar(hf(0))

        to_fit.drop(to_fit[to_fit['ph'] > 0].index, inplace=True)

        to_fit = pd.concat([pd.DataFrame(data={'mag': Zreal_hf, 'ph': 0.0, 'real': Zreal_hf, 'imag': 0.0},
                                index=[1e5], columns=to_fit.columns), to_fit])


    Z11_exp = np.array(to_fit.real.tolist()) + 1j*np.array(to_fit.imag.tolist())

    def residual(scale, Z11_model, Z11_exp):
        '''
        Returns average distance of error between the model and experimental data
        '''
        return (1./len(Z11_model))*np.sqrt(sum((np.real(Z11_exp) - scale*np.real(Z11_model))**2 + (np.imag(Z11_exp) - scale*np.imag(Z11_model))**2))

    Z_array = np.array(Z)
    results_array = np.ndarray((len(Z_array), 3))

    mask = [i for i, f in enumerate(Z.columns) if f in to_fit.index]

    for run, __ in enumerate(Z_array):
        res = minimize(residual, 10.0, args=(Z_array[run, mask], Z11_exp), tol=1e-5)

        results_array[run,0] = run
        results_array[run,1] = res.x # scale
        results_array[run,2] = res.fun # residual

    results = pd.DataFrame(results_array, columns=['run', 'scale', 'residual'])

    sorted_results = results.sort_values(['residual'])
    sorted_results['residual'] = sorted_results['residual'].map(lambda x: x*100/(to_fit['mag'].mean()))

    best_fit =int(sorted_results['run'].iloc[0])
    Z11_model = sorted_results['scale'].iloc[0]*Z.iloc[best_fit, mask]

    fit = zip(Z11_model.index, Z11_model.map(np.real), Z11_model.map(np.imag))
    return fit, sorted_results.iloc[0:100]

def calculateRsquared(ydata, ymodel):
    """ Returns the coefficient of determination (:math:`R^2`)

    Parameters
    -----------------
    ydata : numpy array
        values of the experimental data
    ymodel : numpy array
        values of the fit model

    Returns
    ------------
    r_squared : float
        the coefficient of determination (:math:`R^2`)

    Notes
    ---------
    :math:`R^2` is calculated as [1]_:

    .. math::

            R^2 = \\frac{\\text{Regression Sum of Squares}}{\\text{Total Sum of Squares}} =
            \\frac{ \\sum_{i=1}^{N} (\hat{y}_i - \\bar{y})^2 }{ \\sum_{i=1}^{N} (y_i - \\bar{y})^2 }

    where :math:`\hat{y}` is the model fit, :math:`y_i` is the experimental data, and :math:`\\bar{y}` is the mean of :math:`y`

    .. [1] Casella, G. & Berger, R. L. Statistical inference. (Thomson Learning, 2002), pp. 556.

    """
    # SStot = (ydata - ydata.mean())**2).sum()
    # SSres =
    return 1 #- SSres/SStot

def fitEquivalentCircuit(data, p0):
    # print("fitting circuit")

    # Define Circuit and Initial Parameters
    global circuit_string

    circuit_string ='s(R1,p(s(R1,W2),E2))'

    f = np.array([run[0] for run in data])
    real = np.array([run[1] for run in data])
    imag = np.array([run[2] for run in data])

    # Calculate best guesses
    r1 = real[np.argmax(f)]
    r2 = real.mean() - r1

    c1 = 1/(f[f>1][np.argmax(np.abs(np.arctan2(imag[f>1],real[f>1])))]*r1)

    parameters = [r1, r2, p0[2], p0[3], c1, p0[5]]


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
    for i, __ in enumerate(covar):
        try:
          error.append(np.absolute(p_cov[i][i])**0.5)
        except:
          error.append(0.0)

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
