import sys
import os
from application import application
from flask import render_template, request, jsonify
from application.fitPhysics import fit_P2D, fit_P2D_by_capacity
import pandas as pd
import numpy as np
import json
from application.ECfit import fitEC


@application.route('/', methods=['GET'])
def index():
    """ Impedance Analyzer Main Page """
    return render_template('index.html', version='20172504')


@application.route('/getExampleData', methods=['GET'])
def getExampleData():
    """ Gets the data from the selected example

    Parameters
    ----------
    filename : str
        filename for selecting example

    Returns
    -------
    data : jsonified data

    """

    filename = request.values["filename"]

    with open('./application/static/data/examples/' + filename, 'r') as f:
        contents = f.read()

    data = to_array(contents)

    return jsonify(data=data)


@application.route('/getUploadData', methods=['POST'])
def getUploadData():
    """ Gets uploaded data """
    f = request.files['data']
    contents = f.read()
    data = to_array(contents)

    return jsonify(data=data)


@application.route('/fitCircuit', methods=['POST'])
def fitCircuit():
    """ fits equivalent circuit

    Parameters
    ----------

    circuit : str
        string defining circuit to fit

    data : str
        string of data of comma separated values of freq, real, imag

    p0 : str
        comma deliminated string of initial parameter guesses

    Returns
    -------

    names : list of strings
        list of parameter names

    values : list of strings
        list of parameter values

    errors : list of strings
        list of parameter errors

    ecFit :


    Notes
    -----


    """
    data = request.values["data"].split(',')
    f = [float(f) for f in data[::3]]
    real = [float(r) for r in data[1::3]]
    imag = [float(i) for i in data[2::3]]
    data = list(zip(f, real, imag))

    circuit = request.values["circuit"]
    p0 = request.values["p0"]
    p0 = [float(p) for p in p0.split(',')]

    p_results, p_error, ecFit = fitEC.equivalent_circuit(data, circuit, p0)

    elements = circuit.replace(',', '-').replace('/', '-').split('-')
    replace = {ord(x): '' for x in ['(', ')', 'p']}
    names = [e.translate(replace) for e in elements]

    return jsonify(names=names,
                   values=p_results.tolist(),
                   errors=p_error,
                   ecFit=ecFit)


@application.route('/fitPhysics', methods=['POST'])
def fitPhysics():
    """ fits physics model

    Parameters
    ----------

    request.values["data"] : string
        comma-separated data string

    Returns
    -------

    fit : list
        list of tuples containing the (f, Zr, Zi) of the best fit P2D model

    names : list
        list of parameter names

    units : list


    values=values,

    errors=errors,

    results=p2d_residuals,

    simulations=p2d_simulations,

    fit_points :

    """
    data = request.values["data"].split(',')
    f = [float(f) for f in data[::3]]
    real = [float(r) for r in data[1::3]]
    imag = [float(i) for i in data[2::3]]
    data = list(zip(f, real, imag))

    fit_type = request.values["fit-type"]

    if fit_type == "cap_contact":
        fit_mAh = float(request.values["fit-mAh"])
        fit_points, fit, sorted_results = fit_P2D_by_capacity(data, fit_mAh)
    else:
        fit_points, fit, sorted_results = fit_P2D(data)

    Z = pd.read_pickle('application/static/data/36500-Z.pkl')

    mask = [f for f, r, i in fit]

    freq = [f for f, r, i in data]

    Z = Z.loc[sorted_results['run'].map(int).values, mask]

    full_P = pd.read_csv('./application/static/data/model_runs.txt')
    full_P.index = full_P['run']
    P = full_P.loc[sorted_results['run'].map(int).values]

    to_skip = ['d2Udcp2_neg', 'd2Udcp2_pos', 'd3Udcp3_neg', 'd3Udcp3_pos']

    mask = [c for c in P.columns if c.split('[')[0] not in to_skip]
    P = P.loc[:, mask]

    full_results = []

    for i, spectrum in Z.iterrows():

        parameters = []

        scale = sorted_results['scale'].loc[i]

        parameters.append({"name": "fit parameter",
                           "value": '{:.4e}'.format(scale*1e4)})

        parameters.append({"name": "run",
                           "value": str(P.loc[i, 'run'])})

        for p in P.columns[1:]:
            parameters.append({"name": p,
                               "units": p.split('[')[-1].strip(']'),
                               "value": '{:.4e}'.format(P.loc[i, p])})

        full_results.append({"run": int(i),
                             "freq": Z.columns.values.tolist(),
                             "real": spectrum.apply(np.real).values.tolist(),
                             "imag": spectrum.apply(np.imag).values.tolist(),
                             "parameters": parameters})

    best_fit = sorted_results['run'].iloc[0]
    param_Series = P.loc[best_fit]

    p2d_residuals = sorted_results.values.tolist()

    parameters = []
    parameters.append({"name": "fit parameter", "units": "cm^2",
                       "value": sorted_results['scale'].iloc[0]*1e4,
                       "sensitivity": "x"})

    for i, parameter in enumerate(param_Series.index):
        if parameter.split('[')[0] not in to_skip:
            parameters.append({"name": parameter.split('[')[0],
                               "units": parameter.split('[')[-1].strip("]"),
                               "value": param_Series.iloc[i],
                               "sensitivity": "x"})

    return jsonify(fit=fit, parameters=parameters, results=p2d_residuals,
                   fit_points=fit_points, full_results=full_results)


def to_array(input):
    """ parse strings of data from ajax requests to return

    """

    try:
        input = input.decode("utf-8")
    except AttributeError:
        pass

    input = input.replace('\r\n', ',')
    input = input.replace('\n', ',')
    col0 = [float(x) for x in input.split(',')[0:-1:3]]
    col1 = [float(x) for x in input.split(',')[1:-1:3]]
    col2 = [float(x) for x in input.split(',')[2:-1:3]]

    return list(zip(col0, col1, col2))
