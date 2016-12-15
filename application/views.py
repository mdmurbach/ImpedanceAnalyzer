from __future__ import print_function
from application import application
from flask import render_template, request
from application.fitModels import fitCircuit, fitP2D
import scipy
import sys, os
import pandas as pd
import numpy as np
import json

# main webpage
@application.route('/', methods=['GET', 'POST'])
@application.route('/index', methods=['GET', 'POST'])
def index():

    default_context = {'upload': False, 'data': "", 'ec_parameters': "", 'ecFit': False, 'p2d_parameters': "", 'p2dFit': False, 'p2d_residuals': "", 'p2d_simulations': "", 'p2d_names': ""}

    #### if POST request triggered by form-data button ####
    if request.method == 'POST' and 'data' in request.files:

        fit_equivalent_circuit = 'fitting-ec' in request.values
        fit_p2d = 'fitting-p2d' in request.values

        if fit_equivalent_circuit:
            circuit_string = request.values["circuit_string"]

            p_string = [p for p in circuit_string if p not in 'ps(),-/']

            p0 = []

            for a, b in zip(p_string[::2], p_string[1::2]):
                p0.append(float(request.values[str(a + b)]))

        #### if POST request contains an uploaded file #####
        if request.files['data'].filename != "":
            # print(request.files['data'].filename, file=sys.stderr)
            f = request.files['data']
            contents = f.read()
            uploaded_data = to_array(contents)

            # check if equivalent circuit check box is checked
            if fit_equivalent_circuit:
                p_results, p_error, ecFit = fitCircuit(uploaded_data, circuit_string, p0)

                ec_parameters = [{"name": u"R1",  "value": format(p_results[0], '.4f'), "sensitivity": format(p_error[0], '.4f')},
                                                {"name": u"R2", "value": format(p_results[1], '.4f'), "sensitivity": format(p_error[1], '.4f')},
                                                {"name":u"W1", "value": format(p_results[2], '.4f'), "sensitivity": format(p_error[2], '.4f')},
                                                {"name": u"W2",  "value": format(p_results[3], '.4f'), "sensitivity": format(p_error[3], '.4f')},
                                                {"name": u"CPE1", "value": format(p_results[4], '.4f'), "sensitivity": format(p_error[4], '.4f')},
                                                {"name":u"CPE2", "value": format(p_results[5], '.4f'), "sensitivity": format(p_error[5], '.4f')}]
            else:
                ec_parameters = ""
                ecFit = False

            # check if p2d check box is checked
            if fit_p2d:
                p2dFit, sorted_results  = fitP2D(uploaded_data)

                Z = pd.read_pickle('./application/static/data/17190-Z.pkl')
                Z.index = range(len(Z))

                Z = Z.loc[sorted_results['run'].map(int).values]
                p2d_simulations = pd.DataFrame(columns=['run', 'freq', 'real', 'imag'])

                p2d_simulations['real'] = Z.apply(lambda y: ','.join(y.map(lambda x: str(np.real(x))).values.tolist()), axis=1)
                p2d_simulations['imag'] = Z.apply(lambda y: ','.join(y.map(lambda x: str(np.imag(x))).values.tolist()), axis=1)
                p2d_simulations['freq'] = Z.apply(lambda y: ','.join(Z.columns.map(str)), axis=1)
                p2d_simulations['run'] = Z.index

                parameters=pd.read_csv('./application/static/data/model_runs-full.txt')
                P = parameters.loc[sorted_results['run'].map(int).values]

                p2d_simulations['param'] =P.apply(lambda y: ','.join(y.map(lambda x: str(x)).values.tolist()), axis=1)

                p2d_simulations = p2d_simulations.values.tolist()

                p2d_names = P.columns.values.tolist()

                best_fit = sorted_results['run'].iloc[0]
                param_Series = parameters.loc[best_fit]

                p2d_residuals = sorted_results.values.tolist()

                p2d_parameters = []
                for i, parameter in enumerate(param_Series.index):
                    p2d_parameters.append({'name': parameter.split('[')[0], "value": param_Series.iloc[i], "sensitivity": "x"})
            else:
                p2d_parameters = ""
                p2dFit = False
                p2d_residuals = ""
                p2d_simulations = ""
                p2d_names = ""

            context = {'upload': False, 'data': example_data, 'ec_parameters': ec_parameters, 'ecFit': ecFit, 'p2d_parameters': p2d_parameters, 'p2dFit': p2dFit, 'p2d_residuals': p2d_residuals, 'p2d_simulations': p2d_simulations, 'p2d_names': p2d_names}

            return render_template('index.html', **context)

        #### else if POST request contains a selection from the example dropdown ####
        elif request.values['example'] != "null":

            # get data from POST request
            filename = request.values['example']
            with open('./application/static/data/examples/' + filename, 'r') as f:
                contents = f.read()
            example_data = to_array(contents)

            # check if equivalent circuit check box is checked
            if fit_equivalent_circuit:
                p_results, p_error, ecFit = fitCircuit(example_data, circuit_string, p0)

                parameter_units = request.values["parameter_units"].split(",")

                print(parameter_units, file=sys.stderr)

                ec_parameters = []

                p_string = [p for p in circuit_string if p not in 'ps(),-/']

                for i, (a, b) in enumerate(zip(p_string[::2], p_string[1::2])):
                    ec_parameters.append({"name": str(a+b),
                                                        "units": parameter_units[i],
                                                        "value": format(p_results[i], '.4f'),
                                                        "sensitivity": format(p_error[i], '.4f'),
                                                        "percent_error": format(100*p_error[i]/p_results[i], '.2f')})

            else:
                ec_parameters = ""
                ecFit = False

            # check if p2d check box is checked
            if fit_p2d:
                p2dFit, sorted_results  = fitP2D(example_data)

                Z = pd.read_pickle('./application/static/data/17190-Z.pkl')
                Z.index = range(len(Z))

                Z = Z.loc[sorted_results['run'].map(int).values]
                p2d_simulations = pd.DataFrame(columns=['run', 'freq', 'real', 'imag'])

                p2d_simulations['real'] = Z.apply(lambda y: ','.join(y.map(lambda x: str(np.real(x))).values.tolist()), axis=1)
                p2d_simulations['imag'] = Z.apply(lambda y: ','.join(y.map(lambda x: str(np.imag(x))).values.tolist()), axis=1)
                p2d_simulations['freq'] = Z.apply(lambda y: ','.join(Z.columns.map(str)), axis=1)
                p2d_simulations['run'] = Z.index

                parameters=pd.read_csv('./application/static/data/model_runs-full.txt')
                P = parameters.loc[sorted_results['run'].map(int).values]

                p2d_simulations['param'] =P.apply(lambda y: ','.join(y.map(lambda x: str(x)).values.tolist()), axis=1)

                p2d_simulations = p2d_simulations.values.tolist()

                p2d_names = P.columns.values.tolist()

                best_fit = sorted_results['run'].iloc[0]
                param_Series = parameters.loc[best_fit]

                p2d_residuals = sorted_results.values.tolist()

                p2d_parameters = []
                for i, parameter in enumerate(param_Series.index):
                    p2d_parameters.append({'name': parameter.split('[')[0], "value": param_Series.iloc[i], "sensitivity": "x"})
            else:
                p2d_parameters = ""
                p2dFit = False
                p2d_residuals = ""
                p2d_simulations = ""
                p2d_names = ""

            context = {'upload': False, 'data': example_data, 'ec_parameters': ec_parameters, 'ecFit': ecFit, 'p2d_parameters': p2d_parameters, 'p2dFit': p2dFit, 'p2d_residuals': p2d_residuals, 'p2d_simulations': p2d_simulations, 'p2d_names': p2d_names}

            return render_template('index.html', **context)

    #### initial load + load after "remove file" button ####
    return render_template('index.html', **default_context)

# @application.route('/mseviz')
# def mseviz():
#     return render_template('mse_viz_v4.html')

def to_array(input):
    input = input.replace('\r\n', ',')
    input = input.replace('\n', ',')
    col0 = [float(x) for x in input.split(',')[0:-1:3]]
    col1 = [float(x) for x in input.split(',')[1:-1:3]]
    col2 = [float(x) for x in input.split(',')[2:-1:3]]

    return zip(col0, col1,col2)

def jsonify(names, values):
    d = {}
    for name, value in zip(names, values):
        d[name] = value

    return json.dumps(d)
