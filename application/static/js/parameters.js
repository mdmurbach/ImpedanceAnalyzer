function createParameterTab(id, name, parameters, units, values, errors) {

    $("#grid-parameters ul").append(
        "<li class='nav' role='presentation'>" +
            "<a role='tab' data-toggle='tab' href='#" + "parameters-" + id + "'>" + name +
            "</a>" +
        "</li>"
    );

    var rows = ""


    if(id=='p2d') {

        var formatted_names = ["fit", "run", "l_{neg}[m]", "l_{sep}[m]", "l_{pos}[m]", "R_{p,neg}[m]", "R_{p,pos}[m]", "\\epsilon_{f,neg}[1]", "\\epsilon_{f,pos}[1]", "\\epsilon_{neg}[1]", "\\epsilon_{sep}[1]", "\\epsilon_{pos}[1]", "C_{dl,neg}[{\\mu}F/cm^2]", "C_{dl,pos}[{\\mu}F/cm^2]", "c_0[mol/m^3]"]

        formatted_names = formatted_names.concat(["D[m^2/s]", "D_{s,neg}[m^2/s]", "D_{s,pos}[m^2/s]", "i_{0,neg}[A/m^2]", "i_{0,pos}[A/m^2]", "t_+^0[1]", "\\alpha_{a,neg}[1]", "\\alpha_{a,pos}[1]", "\\kappa_0[S/m]", "\\sigma_{neg}[S/m]", "\\sigma_{pos}[S/m]", "{\\frac{dU}{dc_p}\\bigg|_{neg}}[V*cm^3/mol]", "{\\frac{dU}{dc_p}\\bigg|_{pos}}[V*cm^3/mol]"])

        parameters.forEach(function(p, i) {
            rows = rows +
            "<tr>" +
            "<td>" + "\\[" + formatted_names[i].split('[')[0] + "\\]" + "</td>" +
            "<td>" + "\\[" + units[i] + "\\]" + "</td>" +
            "<td>" + values[i].toPrecision(4) + "</td>" +
            "</tr>"
        });
    } else {
        parameters.forEach(function(p, i) {
            rows = rows +
            "<tr>" +
            "<td>" + "\\[" + p.split('[')[0] + "\\]" + "</td>" +
            "<td>" + "\\[" + units[i] + "\\]" + "</td>" +
            "<td>" + values[i].toPrecision(4) + "</td>" +
            "<td>" + errors[i].toPrecision(4) + "</td>" +
            "<td>" + (100*errors[i]/values[i]).toPrecision(4)  + "</td>"
            "</tr>"
        });
    }

    // append tab-content
        if(id=='p2d') {
            $("#grid-parameters div.tab-content").append(
                "<div class='tab-pane' id='" + "parameters-" + id + "'>" +
                    "<div class='table-responsive' id='table-div'>"+
                        "<table class='table table-condensed' id='parameter-estimates'>" +
                            "<tr>" +
                                "<th>Parameter</th>" +
                                "<th>Units</th>" +
                                "<th>Best Estimate</th>" +
                            "</tr>" +
                            rows +
                        "</table>" +
                    "</div>" +
                "</div>"
            );
        } else {
            $("#grid-parameters div.tab-content").append(
                "<div class='tab-pane' id='" + "parameters-" + id + "'>" +
                    "<div class='table-responsive' id='table-div'>"+
                        "<table class='table table-condensed' id='parameter-estimates'>" +
                            "<tr>" +
                                "<th>Parameter</th>" +
                                "<th>Units</th>" +
                                "<th>Best Estimate</th>" +
                                "<th>Error<i class='glyphicon glyphicon-question-sign' id='info-parameters' data-toggle='tooltip' title='Additional information' aria-hidden='true'></i></th>" +
                                "<th>% Error</th>" +
                            "</tr>" +
                            rows +
                        "</table>" +
                    "</div>" +
                "</div>"
            );
        }

    $("#grid-parameters ul li:first").addClass('active')
    $("#grid-parameters div.tab-content div.tab-pane:first").addClass('active')
    $('#grid-parameters td').each(function(i,d) { renderMathInElement(d); })

}
