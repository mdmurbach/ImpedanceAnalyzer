from __future__ import print_function
from application import application
from flask import render_template, request, jsonify
from application.fitModels import fitEC, fitP2D_matchHF
import scipy
import sys, os
import pandas as pd
import numpy as np
import json

# main webpage
@application.route('/', methods=['GET'])
@application.route('/index', methods=['GET'])
def index():
    "Impedance Analyzer Main Page"

    return render_template('index.html')

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
    f = request.files['data']
    contents = f.read()
    data = to_array(contents)

    return jsonify(data=data)

@application.route('/fitCircuit', methods=['POST'])
def fitCircuit():
    data = request.values["data"].split(',')
    f = [float(f) for f in data[::3]]
    real = [float(r) for r in data[1::3]]
    imag = [float(i) for i in data[2::3]]
    data = zip(f,real,imag)

    circuit = request.values["circuit"]
    p0 = request.values["p0"]
    p0 = [float(p) for p in p0.split(',')]

    p_results, p_error, ecFit = fitEC(data, circuit, p0)

    names = [param.replace('(','').replace(')','').replace('p','') for param in circuit.replace(',', '-').replace('/', '-').split('-')]
    return jsonify(names=names, values=p_results.tolist(), errors=p_error, ecFit=ecFit)

@application.route('/fitPhysics', methods=['POST'])
def fitPhysics():
    data = request.values["data"].split(',')
    f = [float(f) for f in data[::3]]
    real = [float(r) for r in data[1::3]]
    imag = [float(i) for i in data[2::3]]
    data = zip(f,real,imag)

    p2dFit, sorted_results  = fitP2D_matchHF(data)

    Z = pd.read_pickle('application/static/data/25227-Z.pkl')
    Z.index = range(len(Z))

    mask = [f for f, r, i in p2dFit]

    freq = [f for f, r, i in data]

    Z = Z.loc[sorted_results['run'].map(int).values, Z.columns >= min(freq)]
    p2d_simulations = pd.DataFrame(columns=['run', 'freq', 'real', 'imag'])

    p2d_simulations['real'] = Z.apply(lambda y: ','.join(y.map(lambda x: str(np.real(x))).values.tolist()), axis=1)
    p2d_simulations['imag'] = Z.apply(lambda y: ','.join(y.map(lambda x: str(np.imag(x))).values.tolist()), axis=1)
    p2d_simulations['freq'] = Z.apply(lambda y: ','.join(Z.columns.map(str)), axis=1)
    p2d_simulations['run'] = Z.index

    parameters=pd.read_csv('./application/static/data/model_runs-full.txt')
    P = parameters.loc[sorted_results['run'].map(int).values]

    p2d_simulations['param'] = P.apply(lambda y: str(sorted_results['scale'].loc[int(y['run'])]*1e4) + ',' + ','.join(y.map(lambda x: str(x)).values.tolist()), axis=1)

    p2d_simulations = p2d_simulations.values.tolist()

    p2d_names = ['fit parameter[cm^2]'] + (P.columns.values.tolist())

    best_fit = sorted_results['run'].iloc[0]
    param_Series = parameters.loc[best_fit]

    p2d_residuals = sorted_results.values.tolist()

    p2d_parameters = []
    p2d_parameters.append({"name": "fit parameter", "units": "cm^2",
                            "value": sorted_results['scale'].iloc[0]*1e4, "sensitivity": "x"})

    for i, parameter in enumerate(param_Series.index):
        p2d_parameters.append({"name": parameter.split('[')[0], "units": parameter.split('[')[-1].strip("]"),
                                                "value": param_Series.iloc[i], "sensitivity": "x"})

    names = [x['name'] for x in p2d_parameters]
    units = [x['units'] for x in p2d_parameters]
    values = [x['value'] for x in p2d_parameters]

    return jsonify(pbFit=p2dFit, names=names, units=units, values=values, errors="", results=p2d_residuals, simulations=p2d_simulations)


def to_array(input):
    input = input.replace('\r\n', ',')
    input = input.replace('\n', ',')
    col0 = [float(x) for x in input.split(',')[0:-1:3]]
    col1 = [float(x) for x in input.split(',')[1:-1:3]]
    col2 = [float(x) for x in input.split(',')[2:-1:3]]

    return zip(col0, col1,col2)
