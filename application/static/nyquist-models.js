function nyquistChart(config) {
    console.log("Generating Nyquist Plot...");

    outerWidth = config.width;
    outerHeight = config.height;

    var margin = {top: 20, right: 20, bottom: 80, left: 80};
    var width = outerWidth - margin.left - margin.right;
    var height = outerHeight - margin.top - margin.bottom;

    var xValue = function(d) { return d[0]; }
    var yValue = function(d) { return d[1]; }

    var xScale = d3.scale.linear()
    var yScale = d3.scale.linear()

    var xAxis = d3.svg.axis()
        .scale(xScale)
        .ticks(5)
        .orient('bottom');

    var yAxis = d3.svg.axis()
        .scale(yScale)
        .ticks(5)
        .orient('left');

    function chart(selection) {
        selection.each(function(data) {

            data = data.map(function(d, i) {
                return [xValue.call(data, d, i), yValue.call(data, d, i)];
            });

            max = d3.max( [d3.max(data, function(d, i) { return d[0]} ) - d3.min(data, function(d, i) { return d[0]} ),
                                    d3.max(data, function(d, i) { return d[1]} ) - d3.min(data, function(d, i) { return d[1]} )]);

            xScale
                .range([0, width])
                .domain([ d3.min(data, function(d, i) { return d[0]} ) - max*.1,  d3.min(data, function(d, i) { return d[0]} ) + max*1.2]);

            yScale
                .range([height, 0])
                .domain([0, max*1.3]);

            // Select the svg element, if it exists.
            var svg = d3.select(this).selectAll("svg").data([data]);

            // Otherwise, create the skeletal chart.
            var gEnter = svg.enter().append("svg").append("g");

            gEnter.append("g").attr("class", "x axis");
            gEnter.append("g").attr("class", "y axis");

            // Update the outer dimensions.
            svg
                .attr("width", outerWidth)
                .attr("height", outerHeight);

            // Update the inner dimensions.
            var g = svg.select("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            g.select(".x.axis")
                .attr('transform', 'translate(0,' + height + ')')
                .call(xAxis)

            g.select(".y.axis")
                .attr('transform', 'translate(0,' + 0 + ')')
                .call(yAxis)

            g.append("text")
                .attr("class", "label")
                .attr("transform", "translate(" + (width/2) + " ," +
                                                (height + margin.top + 10) + ")")
                .style("text-anchor", "middle")
                .text("Z_real");

            g.append("text")
                .attr("class", "label")
                .attr("transform", "rotate(-90)")
                .attr("y", 20 - margin.left)
                .attr("x",0 - (height / 2))
                .attr("dy", "1em")
                .style("text-anchor", "middle")
                .text("-Z_imag");

            var circles = g.selectAll("circle").data(data);

            circles.enter().append('circle');
            circles
                .attr("class", "default-circle")
                .attr("cx", function (d) { return xScale( d[0] ); } )
                .attr("cy", function (d) { return yScale( d[1] ); } )
                .attr("r", 5)
                .on("mouseover", function(d,i) {
                    d3.select(this)
                        .classed("selected-circle", true)
                        .attr("r", 10);

                    d3.select("#phase").selectAll("circle")
                        .data(data)
                        .classed("selected-circle", function(d2, i2) {return i2 == i; })
                        .attr("r", function(d2, i2) { if(i2 == i) { return 10; } else { return 5;} });

                    d3.select("#magnitude").selectAll("circle")
                        .data(data)
                        .classed("selected-circle", function(d2, i2) {return i2 == i; })
                        .attr("r", function(d2, i2) { if(i2 == i) { return 10; } else { return 5;} });

                })
                .on("mouseout", function(d,i) {
                    d3.select(this)
                        .attr("class", "default-circle")
                        .attr("r", 5);

                    d3.select("#phase").selectAll("circle")
                        .data(data)
                        .attr("class", "default-circle")
                        .attr("r", 5);

                    d3.select("#magnitude").selectAll("circle")
                        .data(data)
                        .attr("class", "default-circle")
                        .attr("r", 5);
                });

            //  Add model fit if exists
            if(config.ec.length > 0) {

                ec = config.ec;

                console.log('Add EC Model');

                var line = d3.svg.line()
                    .x(function(d) { return xScale( d[1] ); })
                    .y(function(d) { return yScale( -d[2] ); });

                    g.append("path")
                      .datum(ec)
                      .attr("class", "ec-fit")
                      .attr("d", line);
            }
        });
    }

    chart.x = function(_) {
        if (!arguments.length) return xValue;
        xValue = _;
        return chart;
    };

    chart.y = function(_) {
        if (!arguments.length) return yValue;
        yValue = _;
        return chart;
    };

    return chart;
}
