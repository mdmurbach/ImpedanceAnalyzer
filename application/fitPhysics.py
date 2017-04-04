""" Provides functions for fitting equivalent circuit and physics-based models

"""

import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def fit_P2D(data):
    """ Fit physics-based model by matching the hf intercept

    Parameters
    ----------
    data : list of tuples
        list of tuples containing (frequency, real impedance,
        imaginary impedance)

    Returns
    -------
    fit_points : list of tuples
        (frequency, real impedance, imaginary impedance) of points
        used in the fitting of the physics-based model

    model_fit : list of tuples
        (frequency, :math:`Z^\prime`, :math:`Z^{\prime\prime}`) of
        the best fitting model

    results_dataframe : pd.DataFrame


    """

    exp_data = prepare_data(data)

    # read in all of the simulation results
    Z = pd.read_pickle('application/static/data/29000-Z.pkl')

    # find the frequencies that fall within the experimental data and create
    # interpolated, to store interpolated experimental data for fitting
    min_f, max_f = min(exp_data['f']), max(exp_data['f'])
    freq_mask = [f for f in Z.columns if min_f <= f <= max_f]
    freq_mask = sorted(freq_mask, reverse=True)

    to_fit = pd.DataFrame(index=freq_mask, columns=['mag', 'ph'])

    # if the frequency isn't already within 1% of the simulation frequencies,
    # quadratically interpolate the nearest 4 points in the magnitude and phase
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

        hf = interp1d(x, y, kind='quadratic')

        Zreal_hf = np.asscalar(hf(0))

        to_fit.drop(to_fit[to_fit['ph'] > 0].index, inplace=True)

        hf_dict = {'mag': Zreal_hf, 'ph': 0.0,
                   'real': Zreal_hf, 'imag': 0.0}

        hf_df = pd.DataFrame(hf_dict, index=[1e5],
                             columns=to_fit.columns)

        to_fit = pd.concat([hf_df, to_fit])

    else:

        # Cubically extrapolate five highest frequencies to find Z_hf
        x = exp_data['real'].iloc[0:5]
        y = exp_data['imag'].iloc[0:5]

        fit = np.polyfit(x, -y, 3)
        func = np.poly1d(fit)

        Zreal_hf = np.real(func.r[np.real(func.r) < min(x)])

        hf_dict = {'mag': Zreal_hf, 'ph': 0.0,
                   'real': Zreal_hf, 'imag': 0.0}

        hf_df = pd.DataFrame(hf_dict, index=[1e5],
                             columns=to_fit.columns)

        to_fit = pd.concat([hf_df, to_fit])

    hf_real = Z.loc[:, 1e5].map(np.real)

    scale = hf_real/Zreal_hf  # m^2
    scale.index = range(1, len(scale)+1)

    Z_data_r = np.array(to_fit.real.tolist())
    Z_data_i = 1j*np.array(to_fit.imag.tolist())

    Z_data = Z_data_r + Z_data_i

    mask = [i for i, f in enumerate(Z.columns) if f in to_fit.index]

    results_array = np.ndarray(shape=(len(Z), 3))

    for run, impedance in enumerate(Z.iloc[:, mask].values):
        scaled = impedance/scale.values[run]

        real_squared = (np.real(Z_data) - np.real(scaled))**2
        imag_squared = (np.imag(Z_data) - np.imag(scaled))**2
        sum_of_squares = sum(np.sqrt(real_squared + imag_squared))

        residual = 1./len(scaled)*sum_of_squares

        results_array[run, 0] = run
        results_array[run, 1] = scale.values[run]
        results_array[run, 2] = residual

    results = pd.DataFrame(results_array, columns=['run', 'scale', 'residual'])

    sorted_results = results.sort_values(['residual'])

    def norm_residual(x):
        return x*100/(to_fit['mag'].mean())

    sorted_results['residual'] = sorted_results['residual'].map(norm_residual)

    best_fit_idx = int(sorted_results['run'].iloc[0])
    best_Z = Z.iloc[best_fit_idx, mask]/sorted_results['scale'].iloc[0]

    points = list(zip(to_fit.index, to_fit.real, to_fit.imag))
    fit = list(zip(best_Z.index, best_Z.map(np.real), best_Z.map(np.imag)))
    results = sorted_results.iloc[0:200]

    return points, fit, results


def prepare_data(data):
    def magnitude(x):
        return np.sqrt(x['real']**2 + x['imag']**2)

    def phase(x):
        return np.arctan2(x['imag'], x['real'])

    # take incoming list of tuples and create a dataframe
    exp_data = pd.DataFrame(data, columns=['f', 'real', 'imag'])
    exp_data['mag'] = exp_data.apply(magnitude, axis=1)
    exp_data['phase'] = exp_data.apply(phase, axis=1)

    # sort from high to low frequencies
    exp_data.sort_values(by='f', ascending=False, inplace=True)
    exp_data.index = range(len(exp_data))

    return exp_data
