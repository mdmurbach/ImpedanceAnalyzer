/* ensure plot button is disabled until file is selected */
$('#selection-form_input-data').on('change', function() {
    $('#selection-form_upload-btn').val($(this).val());
    $('#plot_btn').prop('disabled', false);
    $('#modelFieldset').removeProp('disabled');
});
$('#exampleselect').on('change', function() {
    $('#plot_btn').prop('disabled', false);
    $('#modelFieldset').removeProp('disabled');
});


/* remove file from upload form */
function removeFile() {
    $('#selection-form_input-data')[0].value = "";
    $('#selection-form_upload-btn')[0].value = "Browse...";
    var exampleSelected = $('#exampleselect option:selected')[0].value != "null"
    if(!exampleSelected) {
        $('#plot_btn').prop('disabled', true);
    }
}

/* control submission of data. Called on click of #plot_btn */
function submitAnalysis() {

    $('#grid-impedance div#bode').empty()
    $('#grid-impedance div#nyquist').empty()
    $('#grid-parameters ul').empty()
    $("#grid-parameters div.tab-content").empty()
    $('#explore-residuals').empty()
    $('#explore-nyquist').empty()

    var exampleSelected = $('#exampleselect option:selected')[0].value != "null"

    var uploadSelected = $('#selection-form_input-data')[0].files.length > 0;

    if(uploadSelected) {
        var formData = new FormData()
        var fileData = $('#selection-form_input-data')[0].files[0]
        formData.append('data', fileData);
        $.ajax({
            url: $SCRIPT_ROOT + '/getUploadData',
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            async: false,
            cache: false,
            success: function(response) {
                initializeCharts(response.data);
                var results = plotModels(response.data);
            },
            error: function(error) {
                console.log(error);
            }
        });
    } else if(exampleSelected){
        var filename = $('#exampleselect option:selected')[0].value;
        $.getJSON($SCRIPT_ROOT + '/getExampleData', {
            data_type: "example",
            filename: filename
        },function(response) {
            initializeCharts(response.data);
            var results = plotModels(response.data);
        }
    );
};
}

/* gets models from the tab pane on the EC popup modal-body,
calls makeRequest() for each, and returns results in an array. */
function plotModels(data) {

    var models = $('#modal-analysis div.modal-body #ECtab .tab-pane').toArray()

    results = []

    while(models.length > 0) {
        model = models.shift();
        var result = makeRequest(model, data);
        results.push(result);
    }

    return results
};

/* handles the sending of GET/POST requests for the given model and data to fit.
Returns the results */
function makeRequest(model, data) {
    var formData = new FormData();

    var id = model.getAttribute('id')
    var name = model.getAttribute('name')
    var modelType = model.getAttribute('model-type')

    var example = 'true'
    var filename = $('#exampleselect option:selected')[0].value

    var result = ""

    if (modelType == "ec") {

        var circuit = model.getAttribute('circuit')
        var units = model.getAttribute('units')

        var inputs = $('#modal-analysis div.modal-body #ECtab .tab-pane#' + id + " :input:not(:button)")

        var p0 = []

        inputs.each(function(i,p) {
            p0[i] = p.value;
        })

        formData.append('p0', p0.toString())
        formData.append('data', data)

        formData.append('filename', filename)
        formData.append('circuit', circuit)

        $.ajax({
            url: $SCRIPT_ROOT + '/fitCircuit',
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            async: false,
            cache: false,
            success: function(response) {
                addData(response.ecFit, id, name, data);
                createParameterTab(id, name, response.names, units.split(','), response.values, response.errors);
                show_download();
                result =  { model: name,
                            names: response.names,
                            units: units.split(','),
                            values: response.values,
                            errors: response.errors};
            },
            error: function(error) {
                console.log(error);
            }
        });
    } else if (modelType == "pb") {

        formData.append('data', data)
        formData.append('filename', filename)

        $.ajax({
            url: $SCRIPT_ROOT + '/fitPhysics',
            type: "POST",
            data: formData,
            async: false,
            processData: false,
            contentType: false,
            cache: false,
            success: function(response) {

                console.log(response.full_results);

                var fit = response.fit
                var fit_points = response.fit_points
                var parameters = response.parameters

                var names = parameters.map(function(d) {return d.name})
                var units = parameters.map(function(d) {return d.units})
                var values = parameters.map(function(d) {return d.value})

                var results = response.results
                var full_results = response.full_results

                addData(fit, id, name, data, fit_points);
                createParameterTab(id, name, names,  units, values, '');
                populateModal(results, full_results, names, data, fit_points);
                addP2dexploreButton();

                result =  { model: name,
                            names: names,
                            units: units,
                            values: values,
                            errors: ''};

                show_download();
            },
            error: function(error) {
                console.log(error);
            }
        });
    }
    return result
}
