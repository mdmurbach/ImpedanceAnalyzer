""" Provides functions for fitting equivalent circuit and physics-based models

"""

import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import leastsq


def fit_P2D(data_string):
    """ Fit physics-based model by matching the hf intercept

    Parameters
    ----------

    data : list of tuples
        (frequency, real impedance, imaginary impedance) of the
        experimental data to be fit

    Returns
    -------

    fit_points : list of tuples
        (frequency, real impedance, imaginary impedance) of points
        used in the fitting of the physics-based model

    best_fit : list of tuples
        (frequency, real impedance, imaginary impedance) of
        the best fitting model

    full_results : pd.DataFrame
        DataFrame of top fits sorted by their residual


    """

    # transform data from string to pd.DataFrame
    data = prepare_data(data_string)

    # read in all of the simulation results
    Z = pd.read_pickle('./application/static/data/33500-Z.pkl')

    # interpolate data to match simulated frequencies
    points_to_fit = interpolate_points(data, Z.columns)

    # find the high frequency real intercept
    Zreal_hf, points_to_fit = find_hf_crossover(data, points_to_fit)

    # scale by matching the high frequency intercept for data and simulation
    hf_real = Z.loc[:, 1e5].map(np.real)
    scale = hf_real/Zreal_hf  # m^2
    scale.index = range(1, len(scale)+1)

    Z_data_r = np.array(points_to_fit['real'].tolist())
    Z_data_i = 1j*np.array(points_to_fit['imag'].tolist())
    Z_data = Z_data_r + Z_data_i

    mask = [i for i, f in enumerate(Z.columns) if f in points_to_fit.index]

    results_array = np.ndarray(shape=(len(Z), 3))

    for run, impedance in enumerate(Z.iloc[:, mask].values):
        scaled = impedance/scale.values[run]

        real_squared = (np.real(Z_data) - np.real(scaled))**2
        imag_squared = (np.imag(Z_data) - np.imag(scaled))**2
        sum_of_squares = sum(np.sqrt(real_squared + imag_squared))

        avg_error = 1./len(scaled)*sum_of_squares/points_to_fit['mag'].mean()

        results_array[run, 0] = run + 1
        results_array[run, 1] = scale.values[run]
        results_array[run, 2] = avg_error*100  # percentage

    results = pd.DataFrame(results_array, columns=['run', 'scale', 'residual'])
    results.index = results['run']

    sorted_results = results.sort_values(['residual'])

    best_fit_idx = int(sorted_results['run'].iloc[0])
    best_Z = Z.loc[best_fit_idx].iloc[mask]/sorted_results['scale'].iloc[0]

    fit_points = list(zip(points_to_fit.index,
                      points_to_fit.real,
                      points_to_fit.imag))

    best_fit = list(zip(best_Z.index,
                        best_Z.map(np.real),
                        best_Z.map(np.imag)))

    NUM_RESULTS = 100

    return fit_points, best_fit, sorted_results.iloc[0:NUM_RESULTS]


def prepare_data(data):
    """ Prepares the experimental data for fitting

    Parameters
    ----------

    data : list of tuples
        experimental impedance spectrum given as list of
        (frequency, real impedance, imaginary impedance)

    Returns
    -------

    data_df : pd.DataFrame
        sorted DataFrame with f, real, imag, mag, and phase columns and
    """

    exp_data = pd.DataFrame(data, columns=['f', 'real', 'imag'])
    exp_data['mag'] = exp_data.apply(magnitude, axis=1)
    exp_data['phase'] = exp_data.apply(phase, axis=1)

    # sort from high to low frequencies
    exp_data.sort_values(by='f', ascending=False, inplace=True)
    exp_data.index = range(len(exp_data))

    return exp_data


def interpolate_points(data, exp_freq):
    """ Interpolates experimental data to the simulated frequencies

    Parameters
    ----------

    data : pd.DataFrame

    """

    # find the frequencies that fall within the experimental data and create
    # interpolated, to store interpolated experimental data for fitting
    min_f, max_f = min(data['f']), max(data['f'])
    freq_mask = [f for f in exp_freq if min_f <= f <= max_f]
    freq_mask = sorted(freq_mask, reverse=True)

    points_to_fit = pd.DataFrame(index=freq_mask, columns=['mag', 'ph'])

    # if the frequency isn't already within 1% of the simulation frequencies,
    # quadratically interpolate the nearest 4 points in the magnitude and phase
    for frequency in points_to_fit.index:
        exact = data[data['f'].between(.99*frequency, 1.01*frequency)]
        if not exact.empty:
            points_to_fit.loc[frequency, 'mag'] = np.asscalar(exact['mag'])
            points_to_fit.loc[frequency, 'ph'] = np.asscalar(exact['phase'])
        else:
            idx = np.argmin(np.abs(frequency - data['f']))

            x = data['f'].iloc[idx-2:idx+3]
            y_mag = data['mag'].iloc[idx-2:idx+3]
            y_phase = data['phase'].iloc[idx-2:idx+3]

            mag = interp1d(x, y_mag, kind='quadratic')
            phase = interp1d(x, y_phase, kind='quadratic')

            points_to_fit.loc[frequency, 'mag'] = mag(frequency)
            points_to_fit.loc[frequency, 'ph'] = phase(frequency)

    points_to_fit['real'] = points_to_fit.mag*(points_to_fit.ph.map(np.cos))
    points_to_fit['imag'] = points_to_fit.mag*(points_to_fit.ph.map(np.sin))

    return points_to_fit


def find_hf_crossover(data, points_to_fit):
    crossover = data[data['imag'] > 0]

    if crossover.index.tolist():
        index = crossover.index.tolist()[-1]

        x = data['imag'].loc[index-2:index+3]
        y = data['real'].loc[index-2:index+3]

        hf = interp1d(x, y, kind='quadratic')

        Zreal_hf = np.asscalar(hf(0))

        positive_Zimag = points_to_fit[points_to_fit['ph'] > 0]

        points_to_fit.drop(positive_Zimag.index, inplace=True)

        hf_dict = {'mag': Zreal_hf, 'ph': 0.0,
                   'real': Zreal_hf, 'imag': 0.0}

        hf_df = pd.DataFrame(hf_dict, index=[1e5],
                             columns=points_to_fit.columns)

        points_to_fit = pd.concat([hf_df, points_to_fit])

    elif max(data['f']) < 1e5:
        # Cubically extrapolate five highest frequencies to find Z_hf
        x = data['real'].iloc[0:5]
        y = data['imag'].iloc[0:5]

        fit = np.polyfit(x, -y, 4)
        func = np.poly1d(fit)

        Zreal_hf = np.real(func.r[np.real(func.r) < min(x)])

        hf_dict = {'mag': Zreal_hf, 'ph': 0.0,
                   'real': Zreal_hf, 'imag': 0.0}

        hf_df = pd.DataFrame(hf_dict, index=[1e5],
                             columns=points_to_fit.columns)

        points_to_fit = pd.concat([hf_df, points_to_fit])

    else:
        Zreal_hf = np.real(data[data['f'] == 1e5]['real'])

    return Zreal_hf, points_to_fit


def magnitude(x):
    return np.sqrt(x['real']**2 + x['imag']**2)


def phase(x):
    return np.arctan2(x['imag'], x['real'])


def fit_P2D_by_capacity(data_string, target_capacity):
    """ Fit physics-based model by matching the capacity and then sliding along real (contact resistance)

    Parameters
    ----------

    data : list of tuples
        (frequency, real impedance, imaginary impedance) of the
        experimental data to be fit

    Returns
    -------

    fit_points : list of tuples
        (frequency, real impedance, imaginary impedance) of points
        used in the fitting of the physics-based model

    best_fit : list of tuples
        (frequency, real impedance, imaginary impedance) of
        the best fitting model

    full_results : pd.DataFrame
        DataFrame of top fits sorted by their residual


    """

    # transform data from string to pd.DataFrame
    data = prepare_data(data_string)

    # read in all of the simulation results
    Z = pd.read_pickle('./application/static/data/38800-Z.pkl')

    # interpolate data to match simulated frequencies
    points_to_fit = interpolate_points(data, Z.columns)

    # find the high frequency real intercept
    # Zreal_hf, points_to_fit = find_hf_crossover(data, points_to_fit)

    Z_data_r = np.array(points_to_fit['real'].tolist())
    Z_data_i = 1j*np.array(points_to_fit['imag'].tolist())
    Z_data = Z_data_r + Z_data_i

    mask = [i for i, f in enumerate(Z.columns) if f in points_to_fit.index]

    results_array = np.ndarray(shape=(len(Z), 4))

    P = pd.read_csv('./application/static/data/model_runs.txt')

    ah_per_v = {'pos': 550*10**6, 'neg': 400*10**6}  # mAh/m^3 - Nitta (2015)

    def scale_by_capacity(d, target_capacity, ah_per_v):
        """ returns the area (cm^2) for the parameter Series capacity
        to match the target capacity

        """

        l_pos, l_neg = d[3], d[1]

        e_pos, e_neg = d[10], d[8]

        e_f_pos, e_f_neg = d[7], d[6]

        area_pos = target_capacity/(ah_per_v['pos']*l_pos*(1-e_pos-e_f_pos))
        area_neg = target_capacity/(ah_per_v['neg']*l_neg*(1-e_neg-e_f_neg))

        return max([area_pos, area_neg])

    area = np.ndarray((len(P), 1))
    for i, p in enumerate(P.values):
        area[i] = scale_by_capacity(p, target_capacity, ah_per_v)

    def contact_residual(contact_resistance, Z_model, Z_data):
        Zr = np.real(Z_model) + contact_resistance - np.real(Z_data)
        Zi = np.imag(Z_model) - np.imag(Z_data)

        return np.concatenate((Zr, Zi))

    avg_mag = points_to_fit['mag'].mean()
    for run, impedance in enumerate(Z.values[:, mask]):
        scaled = impedance/area[run]

        p_values = leastsq(contact_residual, 0, args=(scaled, Z_data))

        contact_resistance = p_values[0]
        shifted = scaled + contact_resistance

        real_squared = (np.real(Z_data) - np.real(shifted))**2
        imag_squared = (np.imag(Z_data) - np.imag(shifted))**2
        sum_of_squares = (np.sqrt(real_squared + imag_squared)).sum()

        avg_error = 100./len(shifted)*sum_of_squares/avg_mag

        results_array[run, 0] = run + 1  # run is 1-indexed
        results_array[run, 1] = area[run]  # m^2
        results_array[run, 2] = avg_error  # percentage
        results_array[run, 3] = contact_resistance  # Ohms

    results = pd.DataFrame(results_array,
                           columns=['run',
                                    'area',
                                    'residual',
                                    'contact_resistance'])

    results.index = results['run']

    # remove contact resistances below zero
    results = results[results['contact_resistance'] > 0]

    sorted_results = results.sort_values(['residual'])

    best_fit_idx = int(sorted_results['run'].iloc[0])
    best_fit_Z = Z.loc[best_fit_idx].iloc[mask]
    best_fit_cr = sorted_results['contact_resistance'].iloc[0]
    best_fit_area = sorted_results['area'].iloc[0]
    best_Z = best_fit_Z/best_fit_area + best_fit_cr

    fit_points = list(zip(points_to_fit.index,
                      points_to_fit.real,
                      points_to_fit.imag))

    best_fit = list(zip(best_Z.index,
                        best_Z.map(np.real),
                        best_Z.map(np.imag)))

    NUM_RESULTS = 50

    return fit_points, best_fit, sorted_results.iloc[0:NUM_RESULTS]
