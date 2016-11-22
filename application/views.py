from __future__ import print_function
from application import application
from flask import render_template, request
from application.fitModels import fitEquivalentCircuit
import scipy
import sys, os

# main webpage
@application.route('/', methods=['GET', 'POST'])
@application.route('/index', methods=['GET', 'POST'])
def index():

    parameter_results = ""
    #### if POST request triggered by form-data button ####
    if request.method == 'POST' and 'data' in request.files:

        fit_equivalent_circuit = 'fitting-ec' in request.values

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

                parameter_results = [{"name": u"R1",  "value": p_results[0], "sensitivity": 7},
                                                {"name": u"R2", "value": p_results[1], "sensitivity": 1.3},
                                                {"name":u"W1", "value": p_results[2], "sensitivity": 3},
                                                {"name": u"W2",  "value": p_results[3], "sensitivity": 7},
                                                {"name": u"CPE1", "value": p_results[4], "sensitivity": 1.3},
                                                {"name":u"CPE2", "value": p_results[5], "sensitivity": 3}]
            else:
                ecFit = False


            return render_template('index.html', chart_title=request.files['data'].filename, upload=True, data=array, parameter_results=parameter_results, ecFit=ecFit)

        #### else if POST request contains a selection from the example dropdown ####
        elif request.values['example'] != "null":

            # get data from POST request
            filename = request.values['example']
            with open('./application/static/data/examples/' + filename, 'r') as f:
                contents = f.read()
            array = to_array(contents)

            # check if equivalent circuit check box is checked
            if fit_equivalent_circuit:
                p_results, ecFit = fitEquivalentCircuit(array, p0)

                parameter_results = [{"name": u"R1",  "value": p_results[0], "sensitivity": 7},
                                                {"name": u"R2", "value": p_results[1], "sensitivity": 1.3},
                                                {"name":u"W1", "value": p_results[2], "sensitivity": 3},
                                                {"name": u"W2",  "value": p_results[3], "sensitivity": 7},
                                                {"name": u"CPE1", "value": p_results[4], "sensitivity": 1.3},
                                                {"name":u"CPE2", "value": p_results[5], "sensitivity": 3}]
            else:
                ecFit = False

            print(ecFit, file=sys.stderr)
            return render_template('index.html', chart_title=filename, upload=False, data=array, parameter_results=parameter_results, ecFit=ecFit)

    #### initial load + load after "remove file" button ####
    return render_template('index.html', chart_title="Welcome", upload=False, data="", parameter_results=parameter_results, ecFit=False)


def to_array(input):
    input = input.replace('\r\n', ',')
    input = input.replace('\n', ',')
    col0 = [float(x) for x in input.split(',')[0:-1:3]]
    col1 = [float(x) for x in input.split(',')[1:-1:3]]
    col2 = [float(x) for x in input.split(',')[2:-1:3]]

    return zip(col0, col1,col2)
