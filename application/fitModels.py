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
    Z = pd.read_pickle('application/static/data/19203-Z.pkl')

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

    best_fit = int(sorted_results['run'].iloc[0])
    Z11_model = sorted_results['scale'].iloc[0]*Z.iloc[best_fit, mask]

    fit = zip(Z11_model.index, Z11_model.map(np.real), Z11_model.map(np.imag))
    return fit, sorted_results.iloc[0:100]

def fitP2D_Rohmic(data):
    """ Fit physics-based model using estimation of ohmic resistance

    Parameters
    -----------------
    data : list of tuples
        list of tuples containing (frequency, real impedance, imaginary impedance)

    """

    # take incoming list of tuples and create a dataframe
    # sorted by descending frequency
    exp_data = pd.DataFrame(data, columns=['f', 'real', 'imag'])
    exp_data['mag'] = exp_data.apply(lambda x: np.sqrt(x[1]**2 + x[2]**2), axis=1)
    exp_data['phase'] = exp_data.apply(lambda x: np.arctan2(x[2], x[1]), axis=1)
    exp_data.sort_values(by='f', ascending=False, inplace=True)
    exp_data.index = range(len(exp_data))

    # read in all of the simulation results
    Z = pd.read_pickle('application/static/data/19203-Z.pkl')

    # find the frequencies that fall within the experimental data and create
    # the DataFrame, to_fit, to store interpolated experimental data for fitting
    min_f, max_f = min(exp_data['f']), max(exp_data['f'])
    freq_mask = sorted([f for f in Z.columns if min_f <= f <= max_f], reverse=True)

    to_fit = pd.DataFrame(index=freq_mask, columns=['mag', 'ph'])

    parameters = pd.read_csv('application/static/data/model_runs-full.txt', nrows = len(Z))

    def calc_hf(p):
        kappaeff_sep = p['kappa_0[S/m]']*p['epsilon_sep[1]']**1.5
        kappaeff_pos = p['kappa_0[S/m]']*p['epsilon_pos[1]']**1.5
        kappaeff_neg = p['kappa_0[S/m]']*p['epsilon_neg[1]']**1.5

        sigmaeff_pos = p['sigma_pos[S/m]']*(1-p['epsilon_pos[1]']-p['epsilon_f_pos[1]'])**1.5
        sigmaeff_neg = p['sigma_neg[S/m]']*(1-p['epsilon_neg[1]']-p['epsilon_f_neg[1]'])**1.5


        R_sep = p['l_sep[m]']/kappaeff_sep
        R_pos = p['l_pos[m]']/(kappaeff_pos + sigmaeff_pos)
        R_neg = p['l_pos[m]']/(kappaeff_neg + sigmaeff_neg)

        R_ohmic = R_sep + R_pos + R_neg

        return R_ohmic

    predicted = parameters.apply(calc_hf, axis=1)

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

    hf_real = Z.loc[:,1e5].map(np.real)

    scale = Zreal_hf/predicted
    scale.index = range(1,len(scale)+1)

    Z11_exp = np.array(to_fit.real.tolist()) + 1j*np.array(to_fit.imag.tolist())

    mask = [i for i, f in enumerate(Z.columns) if f in to_fit.index]

    results_array = np.ndarray(shape=(len(Z),3))

    for run, impedance in enumerate(Z.iloc[:,mask].values):
        scaled = impedance*scale.values[run]

        residual = 1./len(scaled)*np.sqrt(sum((np.real(Z11_exp) - np.real(scaled))**2 + (np.imag(Z11_exp) - np.imag(scaled))**2))

        results_array[run,0] = run
        results_array[run,1] = scale.values[run]
        results_array[run,2] = residual

    results = pd.DataFrame(results_array, columns=['run', 'scale', 'residual'])

    sorted_results = results.sort_values(['residual'])
    sorted_results['residual'] = sorted_results['residual'].map(lambda x: x*100/(to_fit['mag'].mean()))

    best_fit_idx = int(sorted_results['run'].iloc[0])
    Z11_model = sorted_results['scale'].iloc[0]*Z.iloc[best_fit_idx, Z.columns >= min(freq)]

    fit = zip(Z11_model.index, Z11_model.map(np.real), Z11_model.map(np.imag))
    return fit, sorted_results.iloc[0:100]

def fitP2D_matchHF(data):
    """ Fit physics-based model by matching the hf intercept

    Parameters
    -----------------
    data : list of tuples
        list of tuples containing (frequency, real impedance, imaginary impedance)

    """

    # take incoming list of tuples and create a dataframe
    # sorted by descending frequency
    exp_data = pd.DataFrame(data, columns=['f', 'real', 'imag'])
    exp_data['mag'] = exp_data.apply(lambda x: np.sqrt(x[1]**2 + x[2]**2), axis=1)
    exp_data['phase'] = exp_data.apply(lambda x: np.arctan2(x[2], x[1]), axis=1)
    exp_data.sort_values(by='f', ascending=False, inplace=True)
    exp_data.index = range(len(exp_data))

    # read in all of the simulation results
    Z = pd.read_pickle('application/static/data/19203-Z.pkl')

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

    hf_real = Z.loc[:,1e5].map(np.real)

    scale = hf_real/Zreal_hf # m^2
    scale.index = range(1,len(scale)+1)

    Z11_exp = np.array(to_fit.real.tolist()) + 1j*np.array(to_fit.imag.tolist())

    mask = [i for i, f in enumerate(Z.columns) if f in to_fit.index]

    results_array = np.ndarray(shape=(len(Z),3))

    for run, impedance in enumerate(Z.iloc[:,mask].values):
        scaled = impedance/scale.values[run]

        residual = 1./len(scaled)*np.sqrt(sum((np.real(Z11_exp) - np.real(scaled))**2 + (np.imag(Z11_exp) - np.imag(scaled))**2))

        results_array[run,0] = run
        results_array[run,1] = scale.values[run]
        results_array[run,2] = residual

    results = pd.DataFrame(results_array, columns=['run', 'scale', 'residual'])

    sorted_results = results.sort_values(['residual'])
    sorted_results['residual'] = sorted_results['residual'].map(lambda x: x*100/(to_fit['mag'].mean()))

    best_fit_idx = int(sorted_results['run'].iloc[0])
    Z11_model = Z.iloc[best_fit_idx, Z.columns >= min(exp_data['f'])]/sorted_results['scale'].iloc[0]

    fit = zip(Z11_model.index, Z11_model.map(np.real), Z11_model.map(np.imag))
    return fit, sorted_results.iloc[0:100]

def calculateChisquared(Zdata, Zmodel, sigma):
    """ Returns the (:math:`\\chi^2`) goodness of fit statistic

    Parameters
    -----------------
    Zdata : numpy array
        values of the experimental data
    Zmodel : numpy array
        values of the fit model
    sigma : numpy array
        standard deviation of measurement

    Returns
    ------------
    chi_squared : float
        goodness of fit statistic (:math:`\\chi^2`)

    Notes
    ---------
    :math:`\\chi^2` is calculated as [1]_:

    .. math::

            \\chi^2 = \\sum_{i=1}^{N_{dat}} \\frac{ \left(Z_i - \\hat{Z}(x_i | P) \\right)^2 }{ \\sigma^2_i }

    where :math:`\hat{Z}(x_i | P)` is the model fit, :math:`Z_i` is the experimental data, and :math:`\\sigma_i` is the standard deviation of measurement :math:`i`

    .. [1] Orazem, M. E. & Tribollet, B. Electrochemical impedance spectroscopy. (Wiley, 2008).

    """
    return sum(((Zdata - Zmodel)**2)/sigma**2)


def fitEC(data, circuit, initial_guess):
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

    global circuit_string

    circuit_string = circuit

    # f = np.array([run[0] for run in data])
    # real = np.array([run[1] for run in data])
    # imag = np.array([run[2] for run in data])

    # # Calculate best guesses
    # r1 = real[np.argmax(f)]
    # r2 = real.mean() - r1
    #
    # c1 = 1/(f[f>1][np.argmax(np.abs(np.arctan2(imag[f>1],real[f>1])))]*r1)
    #
    # parameters = [r1, r2, initial_guess[2], initial_guess[3], c1, initial_guess[5]]


    freq = np.array([a for a,b,c in data])
    zr = np.array([b for a,b,c in data])
    zi =np.array([c for a,b,c in data])
    zrzi = zr + 1j*zi

    # Simulates Initial Conditions and Performs Least
    # Squares fit of circuit(s)
    # sim_data = compute_circuit(parameters, circuit_string, freq)

    p_values, covar, info, errmsg, ier = leastsq(residuals, initial_guess, args=(zrzi, freq), maxfev=100000,
                                            ftol=1E-13,  full_output=True)

    p_error = []
    if covar != None:
        s_sq = ((residuals(p_values, zrzi, freq)**2).sum())/(len(zrzi) - len(p_values))
        p_cov = covar * s_sq
        for i, __ in enumerate(covar):
            try:
              p_error.append(np.absolute(p_cov[i][i])**0.5)
            except:
              p_error.append(0.0)
    else:
        p_error = len(p_values)*[-1]

    fit_data_1 = computeCircuit(circuit_string, p_values.tolist(), freq.tolist())

    fit_zrzi = [a[1] for a in fit_data_1]

    fit = zip(freq, np.real(fit_zrzi), np.imag(fit_zrzi))

    r_squared = 1

    return p_values, p_error, fit#, r_squared

def residuals(param, y, x):
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

    print(circuit_string, fil=sys.stderr)

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
