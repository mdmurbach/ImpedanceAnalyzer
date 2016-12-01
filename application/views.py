from __future__ import print_function
from application import application
from flask import render_template, request
from application.fitModels import fitEquivalentCircuit, fitP2D
import scipy
import sys, os
import pandas as pd

# main webpage
@application.route('/', methods=['GET', 'POST'])
@application.route('/index', methods=['GET', 'POST'])
def index():

    ec_parameters = ""
    p2d_parameters = ""
    #### if POST request triggered by form-data button ####
    if request.method == 'POST' and 'data' in request.files:

        fit_equivalent_circuit = 'fitting-ec' in request.values
        fit_p2d = 'fitting-p2d' in request.values

        if fit_equivalent_circuit:
            param_string = [request.values['R1'], request.values['R2'],
                                        request.values['W1'], request.values['W2'],
                                        request.values['C1'], request.values['C2']]

            p0 = [float(p) for p in param_string]

            print(p0, file=sys.stderr)

        #### if POST request contains an uploaded file #####
        if request.files['data'].filename != "":
            print(request.files['data'].filename, file=sys.stderr)
            f = request.files['data']
            contents = f.read()
            array = to_array(contents)

            # check if equivalent circuit check box is checked
            if fit_equivalent_circuit:
                p_results, ecFit = fitEquivalentCircuit(array, p0)

                ec_parameters = [{"name": u"R1",  "value": p_results[0], "sensitivity": 7},
                                                {"name": u"R2", "value": p_results[1], "sensitivity": 1.3},
                                                {"name":u"W1", "value": p_results[2], "sensitivity": 3},
                                                {"name": u"W2",  "value": p_results[3], "sensitivity": 7},
                                                {"name": u"CPE1", "value": p_results[4], "sensitivity": 1.3},
                                                {"name":u"CPE2", "value": p_results[5], "sensitivity": 3}]
            else:
                ecFit = False

            # check if p2d check box is checked
            if fit_p2d:
                best_fit, p2dFit = fitP2D(array)

                p2d_parameters = [{"name": u"R1",  "value": p_results[0], "sensitivity": 7},
                                                {"name": u"R2", "value": p_results[1], "sensitivity": 1.3},
                                                {"name":u"W1", "value": p_results[2], "sensitivity": 3},
                                                {"name": u"W2",  "value": p_results[3], "sensitivity": 7},
                                                {"name": u"CPE1", "value": p_results[4], "sensitivity": 1.3},
                                                {"name":u"CPE2", "value": p_results[5], "sensitivity": 3}]
            else:
                p2dFit = False


            return render_template('index.html', chart_title=request.files['data'].filename, upload=True, data=array, ec_parameters=ec_parameters, ecFit=ecFit, p2dFit=p2dFit)

        #### else if POST request contains a selection from the example dropdown ####
        elif request.values['example'] != "null":

            # get data from POST request
            filename = request.values['example']
            with open('./application/static/data/examples/' + filename, 'r') as f:
                contents = f.read()
            array = to_array(contents)

            # check if equivalent circuit check box is checked
            if fit_equivalent_circuit:
                p_results, p_error, ecFit = fitEquivalentCircuit(array, p0)

                ec_parameters = [{"name": u"R1",  "value": format(p_results[0], '.4f'), "sensitivity": format(p_error[0], '.4f')},
                                                {"name": u"R2", "value": format(p_results[1], '.4f'), "sensitivity": format(p_error[1], '.4f')},
                                                {"name":u"W1", "value": format(p_results[2], '.4f'), "sensitivity": format(p_error[2], '.4f')},
                                                {"name": u"W2",  "value": format(p_results[3], '.4f'), "sensitivity": format(p_error[3], '.4f')},
                                                {"name": u"CPE1", "value": format(p_results[4], '.4f'), "sensitivity": format(p_error[4], '.4f')},
                                                {"name":u"CPE2", "value": format(p_results[5], '.4f'), "sensitivity": format(p_error[5], '.4f')}]
            else:
                ecFit = False

            # check if p2d check box is checked
            if fit_p2d:
                best_fit, p2dFit = fitP2D(array)

                parameters=pd.read_csv('./application/static/data/model_runs-full.txt')

                param_Series = parameters.loc[best_fit-1]
                print(param_Series, file=sys.stderr)


                p2d_parameters = []
                for parameter in range(len(param_Series)):
                    p2d_parameters.append({'name': param_Series.index[parameter], "value": param_Series.iloc[parameter], "sensitivity": "x"})
                # p2d_parameters = [{"name": u"R1",  "value": p_results[0], "sensitivity": 7},
                #                                 {"name": u"R2", "value": p_results[1], "sensitivity": 1.3},
                #                                 {"name":u"W1", "value": p_results[2], "sensitivity": 3},
                #                                 {"name": u"W2",  "value": p_results[3], "sensitivity": 7},
                #                                 {"name": u"CPE1", "value": p_results[4], "sensitivity": 1.3},
                #                                 {"name":u"CPE2", "value": p_results[5], "sensitivity": 3}]
            else:
                p2dFit = False

            return render_template('index.html', chart_title=filename, upload=False, data=array, ec_parameters=ec_parameters, ecFit=ecFit, p2d_parameters=p2d_parameters, p2dFit=p2dFit)

    #### initial load + load after "remove file" button ####
    return render_template('index.html', chart_title="Welcome", upload=False, data="", ec_parameters=ec_parameters, ecFit=False, p2d_parameters=p2d_parameters, p2dFit=False)


def to_array(input):
    input = input.replace('\r\n', ',')
    input = input.replace('\n', ',')
    col0 = [float(x) for x in input.split(',')[0:-1:3]]
    col1 = [float(x) for x in input.split(',')[1:-1:3]]
    col2 = [float(x) for x in input.split(',')[2:-1:3]]

    return zip(col0, col1,col2)
