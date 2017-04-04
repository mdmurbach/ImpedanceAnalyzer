function initializeCharts(data) {
    outerWidth = d3.select('#bode').node().getBoundingClientRect().width;
    outerHeight = d3.select('#bode').node().getBoundingClientRect().height;

    if(outerHeight == 0) { outerHeight = 500};

    var margin = {top: outerHeight*.01, right: outerWidth*.01, bottom: outerHeight*.12, left: outerWidth*.12};
    var width = outerWidth - margin.left - margin.right;
    var height = outerHeight/2 - margin.top - margin.bottom;

    nyquist_config = {
        outerWidth: outerWidth,
        outerHeight: outerHeight,
        element: document.querySelector('#nyquist'),
        data: data
    }

    window.nyquist = new Nyquist(nyquist_config);

    bode_config = {
        outerWidth: outerWidth,
        outerHeight: outerHeight,
        element: document.querySelector('#bode'),
        data: data
    }

    window.bode = new Bode(bode_config);

}

function addData(ec, id, name, data, fit_data /* optional */) {

    outerWidth = d3.select('#bode').node().getBoundingClientRect().width;
    outerHeight = d3.select('#bode').node().getBoundingClientRect().height;

    var margin = {top: outerHeight*.01, right: outerWidth*.01, bottom: outerHeight*.12, left: outerWidth*.12};
    var width = outerWidth - margin.left - margin.right;
    var height = outerHeight/2 - margin.top - margin.bottom;

    window.bode.addModel(ec, id + "-fit", name);

    window.nyquist.addModel(ec, "model-fit", id + "-fit");

    if (fit_data) {
        window.nyquist.addPoints(fit_data, "fit_points");
    }

    // add legend
    var paths = []

    $('#bode svg#phase path').not('.domain').each(function(i,d) {
        paths[i] = {name: d.getAttribute('name'),
                    id: d.getAttribute('id').split('-')[0]}
    });

    var legend = d3.select("#nyquist svg").selectAll("circle#legend")
                    .data(paths);

    legend.enter().append('text');

    legend
        .attr("id", "legend")
        .attr('r', 5)
        .attr('x', margin.left + 40)
        .attr('y', function(d,i) {return 40 + 20*(i+1) + "px";})
        .attr('class', function(d) {return d.id + '-legend';})
        .text(function(d) {return d.name;});

}
