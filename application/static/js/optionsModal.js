$('.dropdown-toggle').dropdown()
$('.dropdown-menu').click(function(e) {
    e.stopPropagation();
});
$('#dropdown-analysis').on('hidden.bs.dropdown', function() {
    var selected = $('input[type="checkbox"]:checked');
    // var names = "Plot";
    // selected.each(function(i,d){
    //     names += d.name;
    // }
    // )
    // if (names != "") {
    //     $('#dropdown-btn').text(names);
    // } else {
    //     $('#dropdown-btn').text("Select")
    // }

    optionsModal(selected);
});

function optionsModal(selected) {
    $("#modal-analysis div.modal-body ul").empty();
    $("#modal-analysis div.modal-body #ECtab").empty();
    loadModal(selected);
}

function loadModal(selected) {
    var circuits =
        {"randles":
            {"name": "Randles",
            "id": "randles",
            "type": "ec",
            "image": "randles.png",
            "parameters": ["R_0", "R_1", "C_1", "W_1", "W_2"],
            "values": ["0.01", "0.005", ".1", ".0001", "200"],
            "units": ["Ohms", "Ohms", "F", "Ohms", "Sec"],
            "circuit": "R_0-p(R_1,C_1)-W_1/W_2"
        },
        "randles_cpe":
            {"name": "Randles w/CPE",
            "id": "randles_cpe",
            "type": "ec",
            "image": "randles_cpe.png",
            "parameters": ["R0", "R1", "E1", "E2", "W1", "W2"],
            "values": ["0.01", "0.005", ".1", ".9", ".0001", "200"],
            "units": ["Ohms", "Ohms", "F", "-", "Ohms", "Sec"],
            "circuit": "R_0-p(R_1,E_1/E_2)-W_1/W_2"
        },
        "two_constant_warburg" :
            {"name": "Two Time Constants",
            "id": "two_constant_warburg",
            "type": "ec",
            "image": "two_time_constants.png",
            "parameters": ["R_0", "R_1", "C_1", "R_2", "C_2", "W_1", "W_2"],
            "values": ["0.01", "0.005", ".1", "0.005", ".1", ".0001", "200"],
            "units": ["Ohms", "Ohms", "F", "Ohms", "F", "Ohms", "Sec"],
            "circuit": "R_0-p(R_1,C_1)-p(R_2,C_2)-W_1/W_2"
        }
    };

    var physics =
        {"p2d":
            {"id": "p2d",
            "name": "P2D",
            "type": "pb",
            "image": "p2d.png"
            }
        };

    if (selected.length > 0) {

        selected.each(function(i,d) {
            if (d.name.startsWith('EC')) {
                name = d.name.split('-')[1]
                circuit = circuits[name]
                // append tab
                $("#modal-analysis div.modal-body ul").append(
                    "<li class='nav' role='presentation'>" +
                        "<a role='tab' data-toggle='tab' href='#" + circuit.id + "'>" +
                            "<img src='impedance-application/static/images/" + circuit.image + "' alt='' />" +
                        "</a>" +
                    "</li>"
                );

                // create form groups for ECtabs
                groups = ""

                circuit.parameters.forEach(function(p, i) {
                    groups = groups +
                        "<div class='form-group'>" +
                            "<label for='" + p + "' class='col-sm-2 control-label'>" + '\\[' + p + '\\]' + "</label>" +
                            "<div class='col-sm-8 input-group'>" +
                                "<input type='text' name='" + p + "' value='" + circuit.values[i] + "' class='form-control'>" +
                                "<div class='input-group-addon'>" +
                                circuit.units[i] + "</div>" +
                            "</div>" +
                        "</div>"
                });

                // append tab-content
                $("#modal-analysis div.modal-body #ECtab").append(
                    "<div class='tab-pane' id='" + circuit.id +
                                    "' name= '" + circuit.name +
                                    "' circuit='" + circuit.circuit  +
                                    "' units='"+ circuit.units +
                                    "' model-type='" + circuit.type + "'>" +
                        "<h5>Initial Guesses</h5>" +
                        "<form class='form-horizontal' id='defineECparam'>" +
                            groups +
                            "<div class='form-group'>" +
                                "<button type='button' class='btn btn-primary btn-next pull-right'>Next</button>" +
                            "</div>" +
                        "</form>" +
                    "</div>"
                );
            }

            if (d.name.startsWith('pb')) {
                name = d.name.split('-')[1]
                model = physics[name]

                // append tab
                $("#modal-analysis div.modal-body ul").append(
                    "<li class='nav' role='presentation'>" +
                        "<a role='tab' data-toggle='tab' href='#" + model.id + "'>" +
                            "<img src='impedance-application/static/images/" + model.image + "' alt='' />" +
                        "</a>" +
                    "</li>"
                );

                groups =  "<div class='form-group'>" +
                                "<label for='P2D-fitting' style='padding:15px'>" +
                                    "Select type of fit" +
                                "</label>" +
                                // "<div class='radio'>" +
                                //     "<label>" +
                                //     "<input type='radio' name='fittingRadio' value='hf-intercept' checked>" +
                                //     "High Frequency Intercept" +
                                //     "</label>" +
                                // "</div>" +
                                "<div class='radio'>" +
                                    "<label>" +
                                    "<input type='radio' name='fittingRadio' value='cap_contact' checked>" +
                                    "Capacity and Contact Resistance" +
                                    "</label>" +
                                    "<div class='input-group' style='width:50%'>" +
                                        "<input type='text' class='form-control' name='fittingmAh' placeholder='Capacity (mAh)' onkeypress='return event.keyCode != 13;'>" +
                                        "<div class='input-group-addon'>mAh</div>" +
                                    "</div>" +
                                "</div>" +
                             "</div>"

                groups += "<br></br><h5>Model runs: <emph>38,800</emph></h5>"

                // append tab-content
                $("#modal-analysis div.modal-body #ECtab").append(
                    "<div class='tab-pane' id='" + model.id +
                                        "' name= '" + model.name +
                                        "' model-type='" + model.type + "'>" +
                        "<form class='form-horizontal' id='defineP2Dparam'>" +
                            groups +
                            "<div class='form-group'>" +
                                "<button type='button' class='btn btn-primary btn-next pull-right'>Next</button>" +
                            "</div>" +
                        "</form>" +
                    "</div>"
                );
            }

        });

        $("#modal-analysis div.modal-body .tab-pane:last .btn-next")
            .removeClass('btn-next')
            .addClass('btn-save')
            .attr('data-dismiss', 'modal')
            .text('Finished');

        $('.btn-next').click(function(){
            $('.nav-tabs > .active').next('li').find('a').trigger('click');
        });

        $("#modal-analysis div.modal-body ul li:first").addClass('active')
        $("#modal-analysis div.modal-body #ECtab div.tab-pane:first").addClass('active')

        $('#modal-analysis label').each(function(i,d) { renderMathInElement(d); })

        $('#modal-analysis').modal('show')
    }
}
