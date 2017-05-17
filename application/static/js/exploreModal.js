/* Adds a button (using button.js) to the main Nyquist plot*/
function addP2dexploreButton() {
    let svg = d3.select("#nyquist svg")

    let width = d3.select('#nyquist').node().getBoundingClientRect().width;
    let height = d3.select('#nyquist').node().getBoundingClientRect().height;

    let g = svg.append('g')
        .attr('class', 'button')
        .attr('transform', 'translate(' + width*.7 + ',' + height*.1 + ')')

    let text = g.append('text')
        .text('Explore P2D Fit')

    button()
        .container(g)
        .text(text)
        .count(0)
        .cb(function() { $('#exploreFitModal').modal('show') })();
}

/* B for downloading parameter table */
function downloadParameterTable() {

    let headers = $('#table-exploreModal .table-header th');
    let columns = headers.map((i,d) => d.innerText).get();

    let csv = 'data:text/csv;charset=utf-8,';

    let parameters = ['area[cm^2]', 'run', 'l_neg[m]', 'l_sep[m]', 'l_pos[m]',
       'Rp_neg[m]', 'Rp_pos[m]','epsilon_f_neg[1]', 'epsilon_f_pos[1]',
       'epsilon_neg[1]', 'epsilon_sep[1]', 'epsilon_pos[1]',
       'Cdl_neg[uF/cm^2]', 'Cdl_pos[uF/cm^2]', 'c0[mol/m^3]', 'D[m^2/s]',
       'Ds_neg[m^2/s]', 'Ds_pos[m^2/s]', 'i0_neg[A/m^2]', 'i0_pos[A/m^2]',
       'tP[1]', 'aa_neg[1]', 'aa_pos[1]', 'kappa_0[S/m]', 'sigma_neg[S/m]',
       'sigma_pos[S/m]', 'dUdcp_neg[V*cm^3/mol]', 'dUdcp_pos[V*cm^3/mol]'];

    parameters.forEach(p => csv += p + ',');
    csv = csv.slice(0, -1) + '\n'; // remove last comma

    columns.filter(d => d != 'name' && d!= 'units').forEach((d,i) => {
        console.log(i, d);
        let values = ""
        $('#table-exploreModal tr:not(.table-header)').each(function() { values += $(this).children().eq(i+2)[0].innerText + ',' })

        csv += values.slice(0,-1) + '\n'
    })

    var encodedUri = encodeURI(csv)

    var link = document.createElement('a');
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "data.csv");

    if (typeof link.download != "undefined") {
        document.body.appendChild(link);
        link.click();
        document.body.appendChild(link);
    } else {
        alert('Unable to download using this browser. Please try again using Firefox or Chrome.')
    }
}


/* Creates the residual plot, nyquist plot, and parameter table within the
 modal for exploring the results.

 @param {Array.<Object>} full_results An array containing impedance objects

 */
function populateModal(full_results, names, data, fit_data) {

    let outerWidth = $(window).width()*0.98/3;
    let outerHeight = $(window).height()*0.7;

    let size = d3.min([outerWidth, outerHeight]);

    let svg_res = d3.select('#explore-residuals').append("svg")

    let margin = {top: 10, right: 10, bottom: 60, left: 60};
    let width = size - margin.left - margin.right;
    let height = size - margin.top - margin.bottom;

    let xScale_res = d3.scale.linear()
    let yScale_res = d3.scale.linear()

    let xAxis_res = d3.svg.axis()
        .scale(xScale_res)
        .ticks(5)
        .orient('bottom');

    let yAxis_res = d3.svg.axis()
        .scale(yScale_res)
        .ticks(5)
        .orient('left');

    xScale_res
        .range([0, width])
        .domain(d3.extent(full_results, (d, i) => i ));

    yScale_res
        .range([height, 0])
        .domain(d3.extent(full_results, (d, i) => d.residual ));


    // Update the outer dimensions.
    svg_res
        .attr("width", outerWidth)
        .attr("height", outerHeight);

    let plot_res = svg_res.append("g");

    plot_res.append("g")
        .attr("class", "x axis")
        .attr("width", width)
        .attr("height", height);

    plot_res.append("g")
        .attr("class", "y axis")
        .attr("width", width)
        .attr("height", height);

    let g_res = svg_res.select("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    g_res.select(".x.axis")
        .attr('transform', 'translate(0,' + height + margin.bottom + ')')
        .call(xAxis_res)

    g_res.select(".y.axis")
        .attr('transform', 'translate(0,' + 0 + ')')
        .call(yAxis_res)

    g_res.append("text")
        .attr("class", "label")
        .attr("transform", "translate(" + (width/2) + " ," +
                                        (height + margin.bottom - 10) + ")")
        .style("text-anchor", "middle")
        .text("Number #");

    g_res.append("text")
        .attr("class", "label")
        .attr("transform", "rotate(-90)")
        .attr("y", -margin.left - 0)
        .attr("x",0 - (height / 2))
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("Avg. % Error");

    let div = d3.select("#explore-residuals").append("div")
        .attr("class", "explore-tooltip")
        .style("opacity", 0);

    let residuals = g_res.selectAll("circle").data(full_results);

    let selected = []

    let color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'];
    let color_count = 0;

    residuals.enter().append('circle');
    residuals
        .attr("cx", (d, i) => xScale_res(i))
        .attr("cy", (d) => yScale_res(d.residual))
        .attr("r", 5)
        .attr("fill", (d) => get_accu_color(d, full_results))
        .on("mouseover", function(d, i) {

            impedance = full_results.find((data) => data['run'] == d.run);

            let area = d.area;
            let contact_resistance = d.contact_resistance;

            let ohmicResistance = calcHFAccuracy(impedance);

            // get names/values of parameters
            parameters = parameter_names_units(impedance);

            // add values for currently moused over run
            impedance['parameters'].forEach(function(d,i) {
                let run = impedance['run']
                parameters[i].set(run.toString(), d['value']);
            });

            // add values for selected runs
            selected.forEach(function(selected_run) {
                let run = selected_run['run']
                parameters.forEach((p, i) => {
                    parameters[i].set(run.toString(), selected_run['parameters'][i].value);
                })
            })

            // update parameter table
            updateParameterTable(parameters, selected);

            // plot the impedance as a line
            plot_impedance(impedance, area, contact_resistance, fit_data)

            if (d3.select(this).attr('class') != 'selected-circle') {
                d3.select(this).attr("r", 7);
            }

            let {positive, negative} = calcCapacity(impedance, area)

            div.transition()
                .duration(1200)
                .style("opacity", 1);

            let to_display = [
                {label: 'Rank', value: (i+1)},
                {label: 'MSE', value: d.residual.toPrecision(2)  + '%'},
                {label: 'Run', value: d.run},
                {label: 'Pos. Capacity', value: positive.toPrecision(4)+'mAh'},
                {label: 'Neg. Capacity', value: negative.toPrecision(4)+'mAh'},
                {label: 'Contact Resistance', value: (1000*contact_resistance).toPrecision(3) + 'mOhms'}];

            display_string = ''
            to_display.forEach(d => display_string += d.label + ': ' + d.value + '<br>')

            div.html(display_string)
                .style("left", 0.6*width + "px")
                .style("top", 0.6*height + "px");

        })
        .on("mouseout", function(d) {
            if (d3.select(this).attr('class') != 'selected-circle') {
                d3.select(this).attr("r", 5)
            }
            window.nyquistExplore.clear('.error_lines');
            window.nyquistExplore.clear('.explore-nyquist-path');

            // get names/values of parameters
            let parameters = parameter_names_units(impedance);

            // add values for selected runs
            selected.forEach(function(selected_run) {
                let run = selected_run['run']
                parameters.forEach((param, i) => {
                    parameters[i].set(run.toString(), selected_run['parameters'][i].value);
                })
            })

            // update parameter table
            updateParameterTable(parameters, selected);

            div.transition()
                .delay(500)
                .duration(1200)
                .style("opacity", 0);

        })
        .on("click", function(d, i) {


            if (d3.select(this).attr('class') != 'selected-circle') {
                d3.select(this).attr("class", "selected-circle")
                d3.select(this).attr("id", "run-" + d.run)
                d3.select(this)
                    .attr("r", 7)
                    .style('stroke-width', 3)
                    .style('stroke', color_list[color_count % 6]);

                impedance = full_results.find((data) => data['run'] == d.run);

                let area = impedance.area;
                let contact_resistance = impedance.contact_resistance;
                scaled = []

                impedance['freq'].forEach((d,i) => {
                    scaled.push([impedance['freq'][i],
                                 impedance['real'][i]/area + contact_resistance,
                                 impedance['imag'][i]/area]);
                });

                var selected_parameters = []

                impedance['parameters'].forEach((d,i) => {
                    selected_parameters[i] = {
                        value: d['value']
                    }
                })

                selected.push({ id:  "run-" + d.run, run: d.run, rank: i + 1, data: scaled, color: color_list[color_count % 6], parameters: selected_parameters});

                color_count += 1;

            } else {
                d3.select(this).classed("selected-circle", false)
                d3.select(this)
                    .attr("r", 5)
                    .style('stroke-width', 0)

                var selected_id = d3.select(this).attr('id')

                selected = $.grep(selected, (d) => d.id == selected_id, true);

            }

            window.nyquistExplore.clear(".explore-nyquist-selected")

            selected.forEach(function(d,i) {
                window.nyquistExplore.addModel(d.data, "explore-nyquist-selected", d.id, d.color)
            });

            // get names/values of parameters
            var parameters = parameter_names_units(impedance);

            // add values for selected runs
            selected.forEach(function(selected_run) {
                let run = selected_run['run']
                parameters.forEach((param, i) => {
                    parameters[i].set(run.toString(), selected_run['parameters'][i].value);
                })
            })

            // update parameter table
            updateParameterTable(parameters, selected);

            d3.select("#explore-nyquist svg").selectAll("text#legend").remove();

            let legend = d3.select("#explore-nyquist svg").selectAll("text#legend")
                            .data(selected);

            legend.enter().append('text');

            legend
                .attr("id", "legend")
                .attr('r', 5)
                .attr('x', margin.left + 40)
                .attr('y', (d,i) => 40 + 20*(i+1) + "px")
                .text((d) => "Rank: " + d.rank + "   Run: " + d.run)
                .style('fill', (d) => d.color)
                .on("click", function(d, i) {
                    d3.selectAll("circle#" + d.id).each(function(d, i) {
                        var onClickFunc = d3.select(this).on("click");
                        onClickFunc.apply(this, [d, i]);
                    });
                });

            if(selected.length > 0) {
                $('#btn-downloadTable').removeClass('hidden')
            } else {
                $('#btn-downloadTable').addClass('hidden')
            }

        });

        let legend_text = [{name: 'Good, Pos. Contact Res.', color: '#0571b0'},
                           {name: 'Suspect, Pos. Contact Res.', color: '#ca0020'}]//,
                        //    {name: 'Good, Neg. Contact Res.', color: '#92c5de'},
                        //    {name: 'Suspect, Neg. Contact Res.', color: '#f4a582'}]

        let residual_legend = d3.select("#explore-residuals svg")
                                .selectAll("text#legend")
                                    .data(legend_text);

        residual_legend.enter().append('text');

        residual_legend
            .attr("id", "legend")
            .attr('x', margin.left + 5)
            .attr('y', (d,i) => 5 + 20*(i+1) + "px")
            .text( (d) => d.name)
            .style('fill', (d) => d.color)


    nyquist_config = {
        outerWidth: outerWidth,
        outerHeight: outerHeight,
        element: document.querySelector('#explore-nyquist'),
        data: data
    }

    window.nyquistExplore = new Nyquist(nyquist_config);

    if (fit_data) {
        window.nyquistExplore.addPoints(fit_data, "fit_points");
    }

}

function get_accu_color(d, full_results) {
    impedance = full_results.find((data) => data['run'] == d.run);
    let contact_resistance = d.contact_resistance;

    accuracy = calcHFAccuracy(impedance);

    if(accuracy < 0.15) {
        if (contact_resistance > 0) {
            return "#0571b0" // dark blue
        } else {
            return "#92c5de" // light blue
        }
    } else {
        if (contact_resistance > 0) {
            return "#ca0020" // dark red
        } else {
            return "#f4a582" // light red
        }
    }
}

function parameter_names_units(impedance) {
    var names = ["fit[cm^2]", "run[]", "l_{neg}[m]", "l_{sep}[m]", "l_{pos}[m]", "R_{p,neg}[m]", "R_{p,pos}[m]", "\\epsilon_{f,neg}[1]", "\\epsilon_{f,pos}[1]", "\\epsilon_{neg}[1]", "\\epsilon_{sep}[1]", "\\epsilon_{pos}[1]", "C_{dl,neg}[{\\mu}F/cm^2]", "C_{dl,pos}[{\\mu}F/cm^2]", "c_0[mol/m^3]", "D[m^2/s]", "D_{s,neg}[m^2/s]", "D_{s,pos}[m^2/s]", "i_{0,neg}[A/m^2]", "i_{0,pos}[A/m^2]", "t_+^0[1]", "\\alpha_{a,neg}[1]", "\\alpha_{a,pos}[1]", "\\kappa_0[S/m]", "\\sigma_{neg}[S/m]", "\\sigma_{pos}[S/m]", "{\\frac{dU}{dc_p}\\bigg|_{neg}}[V*cm^3/mol]", "{\\frac{dU}{dc_p}\\bigg|_{pos}}[V*cm^3/mol]"];

    parameters = []

    names.forEach(function(d,i) {
        var p = new Map();

        p.set('name', "\\[" + d.split("[")[0] + "\\]")
        p.set('units', "\\[" + d.split("[")[d.split("[").length-1].replace("]","") + "\\]")

        parameters[i] = p
    })
    return parameters
}

function updateParameterTable(parameters, selected) {

    var columns = Array.from(parameters[0].keys());

    var num_cols = columns.length;
    var col_range = Array.from({length: num_cols}, (v, k) => k+1);

    d3.select("#table-exploreModal tbody .table-header").selectAll('th').remove()

    d3.select("#table-exploreModal tbody .table-header").selectAll('th')
        .data(columns)
        .enter()
        .append("th")
        .html((d) => d);

    d3.select("#table-exploreModal tbody").selectAll(".dataRow")
        .data(parameters)
        .enter()
        .append("tr")
        .attr("class", "dataRow");

    d3.select("#table-exploreModal tbody").selectAll(".dataRow")
        .data(parameters)
        .attr("class", "dataRow");

    d3.selectAll("#table-exploreModal tbody .dataRow")
        .selectAll("td").remove()

    d3.selectAll("#table-exploreModal tbody .dataRow")
        .selectAll("td")
        .data(col_range)
        .enter()
        .append("td")

    d3.selectAll("#table-exploreModal tbody .dataRow")
        .selectAll("td")
        .data(
            function(row) {
                return columns.map(function(d) {
                    return {value: row.get(d)};
            })
        })
        .html((d) => d.value);

    $('#table-exploreModal tbody td').each((i,d) => renderMathInElement(d))

    let selected_cols = d3.select("#table-exploreModal tbody .table-header").selectAll('th').filter(function(d) {return parseInt(d)})[0]

    selected_cols.forEach(function(d,i) {
        let run = parseInt(d.textContent);
        let selected_run = selected.filter(d => d.run == run)[0];
        if(selected_run) {
            $('#exploreFitModal table#table-exploreModal tr td:nth-child(' + (i+3) + ')').css('color', selected_run.color);
        }
    })
};

function plot_impedance(data, area, contact_resistance, fit_data) {

    let impedance = [];

    data['freq'].forEach(function(d,i) {
        impedance[i] = {
            f: +d,
            real: +data['real'][i] + contact_resistance*area,
            imag: -1*(+data['imag'][i])
        }
    })

    window.nyquistExplore.updateModel(impedance, area, "explore-nyquist-path")

    window.nyquistExplore.clear('.error_lines');

    var length = 0;

    impedance.forEach(function(d,i) {
        var bisector = [{real: d.real/area, imag: d.imag/area}, {real: fit_data[i][1], imag: -1*fit_data[i][2]}];
        length += Math.sqrt(Math.pow(d.real/area - fit_data[i][1], 2) + Math.pow(d.imag/area + fit_data[i][2], 2))
        window.nyquistExplore.addLines(bisector, "error_lines")
    });
}

function createParameterTable(element, parameters) {

    d3.select(element).select("#table-exploreModal tbody").selectAll(".dataRow")
        .data(parameters)
        .enter()
        .append("tr")
        .attr("class", "dataRow")

    d3.select("#table-exploreModal tbody").selectAll(".dataRow")
        .data(parameters)
        .attr("class", "dataRow")
        // .classed("info", function(d, i) {
        //     return last_p[i].value != parameters[i].value;
        // })

    d3.selectAll("#table-exploreModal tbody .dataRow")
       .selectAll("td")
       .data([0,1,2])
       .enter()
       .append("td")

   d3.selectAll("#table-exploreModal tbody .dataRow")
      .selectAll("td")
      .data(
          function(row) {
          return ["name", "units", "value"].map(function(d) {
              return {value: row[d]};
          });
      })
      .html(function(d) { return d.value; });
}

function calcOhmicR(impedance) {
    parameters = impedance['parameters']

    l_neg = parameters.find(function(d) { return d.name == "l_neg[m]";}).value
    l_sep = parameters.find(function(d) { return d.name == "l_sep[m]";}).value
    l_pos = parameters.find(function(d) { return d.name == "l_pos[m]";}).value

    epsilon_neg = parameters.find((d) => d.name == "epsilon_neg[1]").value
    epsilon_sep = parameters.find((d) => d.name == "epsilon_sep[1]").value
    epsilon_pos = parameters.find((d) => d.name == "epsilon_pos[1]").value

    epsilon_f_neg = parameters.find((d) => d.name == "epsilon_f_neg[1]").value
    epsilon_f_pos = parameters.find((d) => d.name == "epsilon_f_pos[1]").value

    sigma_neg = parameters.find((d) => d.name == "sigma_neg[S/m]").value
    sigma_pos = parameters.find((d) => d.name == "sigma_pos[S/m]").value

    kappa = parameters.find((d) => d.name == "kappa_0[S/m]").value

    r_sep = l_sep/(kappa*Math.pow(epsilon_sep,4))

    r_pos = l_pos/(kappa*Math.pow(epsilon_pos,4) +      sigma_pos*Math.pow(1-epsilon_pos-epsilon_f_pos, 4))

    r_neg = l_neg/(kappa*Math.pow(epsilon_neg,4) +      sigma_neg*Math.pow(1-epsilon_neg-epsilon_f_neg, 4))

    return r_sep + r_pos + r_neg
}

function calcHFAccuracy(impedance) {

    predicted = calcOhmicR(impedance)

    hf_sim = impedance.real[0]

    return (hf_sim - predicted)/predicted;
}

function calcCapacity(impedance, area) {
    parameters = impedance['parameters']

    l_neg = parameters.find((d) => d.name == "l_neg[m]").value
    l_pos = parameters.find((d) => d.name == "l_pos[m]").value

    epsilon_neg = parameters.find((d) => d.name == "epsilon_neg[1]").value
    epsilon_pos = parameters.find((d) => d.name == "epsilon_pos[1]").value

    epsilon_f_neg = parameters.find((d) => d.name == "epsilon_f_neg[1]").value
    epsilon_f_pos = parameters.find((d) => d.name == "epsilon_f_pos[1]").value

    const volEnerCap_pos = 550*10**6; // mAh/m^3
    const volEnerCap_neg = 400*10**6; // mAh/m^3

    posCapacity = area*l_pos*volEnerCap_pos*(1-epsilon_pos-epsilon_f_pos)
    negCapacity = area*l_neg*volEnerCap_neg*(1-epsilon_neg-epsilon_f_neg)

    return {positive: posCapacity, negative: negCapacity}

}
