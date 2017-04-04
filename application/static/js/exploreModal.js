function addP2dexploreButton() {
    var svg = d3.select("#nyquist svg")

    outerWidth = d3.select('#bode').node().getBoundingClientRect().width;
    outerHeight = d3.select('#bode').node().getBoundingClientRect().height;

    var g = svg.append('g')
        .attr('class', 'button')
        .attr('transform', 'translate(' + outerWidth*.75 + ',' + outerHeight*.1 + ')')

    var text = g.append('text')
        .text('Explore P2D Fit')

    button()
        .container(g)
        .text(text)
        .count(0)
        .cb(function() { $('#exploreFitModal').modal('show') })();
}

var color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'];
var color_count = 0;

function populateModal(sorted_results, simulations, names, data, fit_data) {
    outerWidth = $(window).width()*.98/3;
    outerHeight = $(window).height()*.7;

    var size = d3.min([outerWidth,outerHeight]);

    var svg_res = d3.select('#explore-residuals').append("svg")

    var margin = {top: 10, right: 10, bottom: 60, left: 60};
    var width = size - margin.left - margin.right;
    var height = size - margin.top - margin.bottom;

    var xScale_res = d3.scale.linear()
    var yScale_res = d3.scale.linear()

    var xAxis_res = d3.svg.axis()
        .scale(xScale_res)
        .ticks(5)
        .orient('bottom');

    var yAxis_res = d3.svg.axis()
        .scale(yScale_res)
        .ticks(5)
        .orient('left');

    xScale_res
        .range([0, width])
        .domain(d3.extent(sorted_results, function(d, i) { return i} ));

    yScale_res
        .range([height, 0])
        .domain(d3.extent(sorted_results, function(d, i) { return d[2]} ));


    // Update the outer dimensions.
    svg_res
        .attr("width", outerWidth)
        .attr("height", outerHeight);

    var plot_res = svg_res.append("g");

    plot_res.append("g")
        .attr("class", "x axis")
        .attr("width", width)
        .attr("height", height);

    plot_res.append("g")
        .attr("class", "y axis")
        .attr("width", width)
        .attr("height", height);

    var g_res = svg_res.select("g")
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

    var div = d3.select("#explore-residuals").append("div")
        .attr("class", "explore-tooltip")
        .style("opacity", 0);

    var residuals = g_res.selectAll("circle").data(sorted_results);

    var last_p = names

    var selected = []

    residuals.enter().append('circle');
    residuals
        .attr("class", "default-circle")
        .attr("cx", function (d, i) { return xScale_res(i); } )
        .attr("cy", function (d) { return yScale_res(d[2]); } )
        .attr("r", 5)
        .on("mouseover", function(d, i) {
            impedance = simulations.find( function(data) { return data[0] == d[0]; });
            scale = d[1];

            var formatted_names = ["fit", "run", "a_{neg}[m^{-1}]", "a_{pos}[m^{-1}]", "l_{neg}[m]", "l_{sep}[m]", "l_{pos}[m]", "R_{p,neg}[m]", "R_{p,pos}[m]", "\\epsilon_{f,neg}[1]", "\\epsilon_{f,pos}[1]", "\\epsilon_{neg}[1]", "\\epsilon_{pos}[1]", "C_{dl,neg}[{\\mu}F/cm^2]", "C_{dl,pos}[{\\mu}F/cm^2]", "c_0[mol/m^3]", "D[m^2/s]", "D_{s,neg}[m^2/s]", "D_{s,pos}[m^2/s]", "i_{0,neg}[A/m^2]", "i_{0,pos}[A/m^2]", "t_+^0[1]", "\\alpha_{a,neg}[1]", "\\alpha_{a,pos}[1]", "\\kappa_0[S/m]", "\\sigma_{neg}[S/m]", "\\sigma_{pos}[S/m]", "{\\frac{dU}{dc_p}\\bigg|_{neg}}[V*cm^3/mol]", "{\\frac{dU}{dc_p}\\bigg|_{pos}}[V*cm^3/mol]"];

            parameters = get_parameters(formatted_names)

            plot_impedance(impedance, scale, fit_data)

            if (d3.select(this).attr('class') != 'selected-circle') {
                d3.select(this).attr("class", "hovered-circle")
            }

            div.transition()
                .duration(200)
                .style("opacity", 1);

            div.html(
                'Rank: ' + (i+1) + '<br>' +
                'MSE:  ' + Math.round(d[2] * 1000)/ 1000  + '%'  + '<br>' +
                'Run: '+ d[0])
                .style("left", 0.8*width + "px")
                .style("top", 0.8*height + "px");

            d3.select("#parameter-estimates tbody").selectAll(".dataRow")
                .data(parameters)
                .enter()
                .append("tr")
                .attr("class", "dataRow")

            d3.select("#parameter-estimates tbody").selectAll(".dataRow")
                .data(parameters)
                .attr("class", "dataRow")
                .classed("info", function(d, i) {
                    return last_p[i].value != parameters[i].value;
                })

            d3.selectAll("#parameter-estimates tbody .dataRow")
               .selectAll("td")
               .data([0,1,2])
               .enter()
               .append("td")

           d3.selectAll("#parameter-estimates tbody .dataRow")
              .selectAll("td")
              .data(
                  function(row) {
                  return ["name", "units", "value"].map(function(d) {
                      return {value: row[d]};
                  });
              })
              .html(function(d) { return d.value; });

              last_p = parameters

        $('#parameter-estimates tbody td').each(function(i,d) { renderMathInElement(d); })

        })
        .on("mouseout", function(d) {
            if (d3.select(this).attr('class') != 'selected-circle') {
                d3.select(this).transition().duration(150).attr("class", "default-circle");
            }
            window.nyquistExplore.clear('.error_lines');
            window.nyquistExplore.clear('.explore-nyquist-path');
        })
        .on("click", function(d, i) {


            if (d3.select(this).attr('class') != 'selected-circle') {
                d3.select(this).attr("class", "selected-circle")
                d3.select(this).attr("id", "run-" + d[0])
                d3.select(this).style('fill', color_list[color_count % 6]);

                impedance = simulations.find( function(data) { return data[0] == d[0]; });

                scale = d[1];

                parsed = [];
                impedance[1].split(',').forEach(function(d,j) {
                    parsed[j] = [+d,+impedance[2].split(',')[j],+impedance[3].split(',')[j]]
                })

                scaled = parsed.map(function(d) { return [d[0],d[1]/scale, d[2]/scale]; });

                selected.push({ id:  "run-" + d[0], rank: i + 1, data: scaled, color: color_list[color_count % 6]});

                color_count += 1;

            } else {
                d3.select(this).attr("class", "default-circle")
                d3.select(this).style('fill', '#009688');

                var selected_id = d3.select(this).attr('id')

                selected = $.grep(selected, function(d) { return d.id == selected_id}, true);

            }

            window.nyquistExplore.clear(".explore-nyquist-selected")

            selected.forEach(function(d,i) {
                window.nyquistExplore.addModel(d.data, "explore-nyquist-selected", d.id, d.color)
            });

            d3.select("#explore-nyquist svg").selectAll("text#legend").remove();

            var legend = d3.select("#explore-nyquist svg").selectAll("text#legend")
                            .data(selected);

            legend.enter().append('text');

            legend
                .attr("id", "legend")
                .attr('r', 5)
                .attr('x', margin.left + 40)
                .attr('y', function(d,i) {return 40 + 20*(i+1) + "px";})
                .text(function(d) {return "Rank: " + d.rank + "   Run: " + d.id.split('-')[1];})
                .style('fill', function(d) { return d.color} )
                .on("click", function(d, i) {
                    d3.selectAll("circle#" + d.id).each(function(d, i) {
                        var onClickFunc = d3.select(this).on("click");
                        onClickFunc.apply(this, [d, i]);
                    });
                });

        });

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

function get_parameters(names) {
    parameters = []

    names.forEach(function(d,i) {
        parameters[i] = {
            name: "\\[" + d.split("[")[0] + "\\]",
            units: "\\[" + d.split("[")[d.split("[").length-1].replace("]","") + "\\]",
            value: parseFloat(impedance[4].split(',')[i]).toPrecision(4)
        }
    })
    return parameters
}

function plot_impedance(data, scale, fit_data) {

    impedance = [];

    data[1].split(',').forEach(function(d,i) {
        impedance[i] = {
            f: +d,
            real: +data[2].split(',')[i],
            imag: -1*(+data[3].split(',')[i])
        }
    })

    window.nyquistExplore.updateModel(impedance, scale, "explore-nyquist-path")

    window.nyquistExplore.clear('.error_lines');

    var length = 0;

    impedance.forEach(function(d,i) {
        var bisector = [{real: d.real/scale, imag: d.imag/scale}, {real: fit_data[i][1], imag: -1*fit_data[i][2]}];
        length += Math.sqrt(Math.pow(d.real/scale - fit_data[i][1], 2) + Math.pow(d.imag/scale + fit_data[i][2], 2))
        window.nyquistExplore.addLines(bisector, "error_lines")
    });
}

function createParameterTable(element, parameters) {

    d3.select(element).select("#parameter-estimates tbody").selectAll(".dataRow")
        .data(parameters)
        .enter()
        .append("tr")
        .attr("class", "dataRow")

    d3.select("#parameter-estimates tbody").selectAll(".dataRow")
        .data(parameters)
        .attr("class", "dataRow")
        .classed("info", function(d, i) {
            return last_p[i].value != parameters[i].value;
        })

    d3.selectAll("#parameter-estimates tbody .dataRow")
       .selectAll("td")
       .data([0,1,2])
       .enter()
       .append("td")

   d3.selectAll("#parameter-estimates tbody .dataRow")
      .selectAll("td")
      .data(
          function(row) {
          return ["name", "units", "value"].map(function(d) {
              return {value: row[d]};
          });
      })
      .html(function(d) { return d.value; });
}
