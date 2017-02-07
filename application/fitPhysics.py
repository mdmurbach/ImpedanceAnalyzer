""" Provides functions for fitting equivalent circuit and physics-based models

"""

from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def fit_P2D(data):
    """ Fit physics-based model by matching the hf intercept

    Parameters
    ----------
    data : list of tuples
        list of tuples containing (frequency, real impedance, imaginary impedance)

    Returns
    -------
    fit_exp :

    fit_result :

    sorted_results :

    """

    # take incoming list of tuples and create a dataframe
    # sorted by descending frequency
    exp_data = pd.DataFrame(data, columns=['f', 'real', 'imag'])
    exp_data['mag'] = exp_data.apply(lambda x: np.sqrt(x[1]**2 + x[2]**2), axis=1)
    exp_data['phase'] = exp_data.apply(lambda x: np.arctan2(x[2], x[1]), axis=1)
    exp_data.sort_values(by='f', ascending=False, inplace=True)
    exp_data.index = range(len(exp_data))

    # read in all of the simulation results
    Z = pd.read_pickle('application/static/data/25227-Z.pkl')

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

    else:

        #Linearly extrapolate three highest frequencies to find Z_hf
        x = exp_data['real'].iloc[0:3]
        y = exp_data['imag'].iloc[0:3]

        fit = np.polyfit(x,-y, 2)
        func = np.poly1d(fit)

        Zreal_hf = func.r[np.real(func.r) < min(x)]

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
    exp_data = zip(to_fit.index, to_fit.real, to_fit.imag)

    return exp_data, fit, sorted_results.iloc[0:200]

def calculateChisquared(Zdata, Zmodel, sigma):
    """ Returns the (:math:`\\chi^2`) goodness of fit statistic

    Parameters
    ----------
    Zdata : numpy array
        values of the experimental data
    Zmodel : numpy array
        values of the fit model
    sigma : numpy array
        standard deviation of measurement

    Returns
    -------
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
