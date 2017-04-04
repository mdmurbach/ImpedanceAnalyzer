function phase(real, imag) {
    return Math.atan2(imag,real)*180/Math.PI;
}

function mag(real, imag) {
    return Math.sqrt(Math.pow(imag, 2) + Math.pow(real, 2));
}

var Bode = function(config) {

    this.outerWidth = config.outerWidth;
    this.outerHeight = config.outerHeight;
    this.element = config.element;
    this.data = config.data;

    if(this.outerHeight == 0) { this.outerHeight = 500};

    this.margin = {top: outerHeight*.01, right: outerWidth*.01, bottom: outerHeight*.12, left: outerWidth*.12};
    this.width = this.outerWidth - this.margin.left - this.margin.right;
    this.height = this.outerHeight/2 - this.margin.top - this.margin.bottom;

    this.draw();
}

Bode.prototype.draw = function() {

    var svg_ph = d3.select(this.element).append("svg").attr("id", "phase")

    var svg_mag = d3.select(this.element).append("svg").attr("id", "magnitude")

    var ph_group = svg_ph.append('g');

    svg_ph
        .attr("width", this.outerWidth)
        .attr("height", this.outerHeight/2);

    var mag_group = svg_mag.append('g');

    svg_mag
        .attr("width", this.outerWidth)
        .attr("height", this.outerHeight/2);

    this.plot_ph = ph_group.append('g')
        .attr('transform', 'translate(' + this.margin.left + "," + this.margin.top + ')')

    this.plot_mag = mag_group.append('g')
        .attr('transform', 'translate(' + this.margin.left + "," + this.margin.top + ')')

    this.createScales();
    this.addAxes();
    this.initializeData();
}

Bode.prototype.createScales = function(){

    this.xScale_ph = d3.scale.log()
        .range([0, this.width])
        .domain(d3.extent(this.data, function(d, i) { return d[0]} ));

    this.yScale_ph = d3.scale.linear()
        .range([this.height, 0])
        .domain([0, 1.1*d3.min(this.data, function(d, i) { return phase(d[1], d[2])} )]);

    this.xScale_mag = d3.scale.log()
        .range([0, this.width])
        .domain(d3.extent(this.data, function(d, i) { return d[0]} ));

    this.yScale_mag = d3.scale.log()
        .range([this.height, 0])
        .domain(d3.extent(this.data, function(d, i) { return mag(d[1], d[2])} ))
        .nice();

}

Bode.prototype.addAxes = function() {

    var xAxis_ph = d3.svg.axis()
        .scale(this.xScale_ph)
        .ticks(5)
        .orient('bottom');

    var yAxis_ph = d3.svg.axis()
        .scale(this.yScale_ph)
        .ticks(5)
        .orient('left');

    var xAxis_mag = d3.svg.axis()
        .scale(this.xScale_mag)
        .ticks(5)
        .orient('bottom');

    var yAxis_mag = d3.svg.axis()
        .scale(this.yScale_mag)
        .ticks(5)
        .orient('left');

    this.plot_ph.append('g')
        .attr('class', 'x axis')
        .attr('transform', 'translate(0,' + this.height + ')')
        .call(xAxis_ph)

    this.plot_ph.append('g')
        .attr('class', 'y axis')
        .attr('transform', 'translate(0,' + 0 + ')')
        .call(yAxis_ph)

    this.plot_mag.append('g')
        .attr('class', 'x axis')
        .attr('transform', 'translate(0,' + this.height + ')')
        .call(xAxis_mag)

    this.plot_mag.append('g')
        .attr('class', 'y axis')
        .attr('transform', 'translate(0,' + 0 + ')')
        .call(yAxis_mag)

    this.plot_ph.append("text")
        .attr("class", "label")
        .attr("transform", "translate(" + (this.width/2) + " ," +
                                        (this.height + this.margin.bottom - 10) + ")")
        .style("text-anchor", "middle")
        .text("Frequency [Hz]");

    this.plot_ph.append("text")
        .attr("class", "label")
        .attr("transform", "rotate(-90)")
        .attr("y", -this.margin.left - 1)
        .attr("x",0 - (this.height / 2))
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("-Phase [deg]");

    this.plot_mag.append("text")
        .attr("class", "label")
        .attr("transform", "translate(" + (this.width/2) + " ," +
                                        (this.height + this.margin.bottom - 10) + ")")
        .style("text-anchor", "middle")
        .text("Frequency [Hz]");

    this.plot_mag.append("text")
        .attr("class", "label")
        .attr("transform", "rotate(-90)")
        .attr("y", -this.margin.left - 1)
        .attr("x",0 - (this.height / 2))
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("Magnitude [Ohms]");
}


Bode.prototype.initializeData = function(){
    var _this = this;

    var phases = this.plot_ph.selectAll("circle").data(this.data);

    phases.enter().append('circle');
    phases
        .attr("class", "default-circle")
        .attr("cx", function (d) { return _this.xScale_ph( d[0] ); } )
        .attr("cy", function (d) { return _this.yScale_ph( phase(d[1], d[2]) ); } )
        .attr("r", 5)
        .on("mouseover", function(d,i) {
            d3.select(this)
                .classed("selected-circle", true)
                .attr("r", 10);

            d3.select("#magnitude").selectAll("circle")
                .data(_this.data)
                .classed("selected-circle", function(d2, i2) {return i2 == i; })
                .attr("r", function(d2, i2) { if(i2 == i) { return 10; } else { return 5;} });

            d3.select("#nyquist").selectAll("circle")
                .data(_this.data)
                .classed("selected-circle", function(d2, i2) {return i2 == i; })
                .attr("r", function(d2, i2) { if(i2 == i) { return 10; } else { return 5;} });
        })
        .on("mouseout", function(d,i) {
            d3.select(this)
                .attr("class", "default-circle")
                .attr("r", 5);

            d3.select("#magnitude").selectAll("circle")
                .data(_this.data)
                .attr("class", "default-circle")
                .attr("r", 5);

            d3.select("#nyquist").selectAll("circle")
                .data(_this.data)
                .attr("class", "default-circle")
                .attr("r", 5);
        });

    var mags = this.plot_mag.selectAll("circle").data(this.data);

    mags.enter().append('circle');
    mags
        .attr("class", "default-circle")
        .attr("cx", function (d) { return _this.xScale_mag( d[0] ); } )
        .attr("cy", function (d) { return _this.yScale_mag( mag(d[1], d[2]) );  } )
        .attr("r", 5)
        .on("mouseover", function(d,i) {
            d3.select(this)
                .classed("selected-circle", true)
                .attr("r", 10);

            d3.select("#phase").selectAll("circle")
                .data(_this.data)
                .classed("selected-circle", function(d2, i2) {return i2 == i; })
                .attr("r", function(d2, i2) { if(i2 == i) { return 10; } else { return 5;} });

            d3.select("#nyquist").selectAll("circle")
                .data(_this.data)
                .classed("selected-circle", function(d2, i2) {return i2 == i; })
                .attr("r", function(d2, i2) { if(i2 == i) { return 10; } else { return 5;} });
        })
        .on("mouseout", function(d,i) {
            d3.select(this)
                .attr("class", "default-circle")
                .attr("r", 5);

            d3.select("#phase").selectAll("circle")
                .data(_this.data)
                .attr("class", "default-circle")
                .attr("r", 5);

            d3.select("#nyquist").selectAll("circle")
                .data(_this.data)
                .attr("class", "default-circle")
                .attr("r", 5);
        });

}

Bode.prototype.addModel = function (data, id, name) {

    var _this = this;

    var line_ph = d3.svg.line()
        .x(function(d) { return _this.xScale_ph( d[0] ); })
        .y(function(d) { return _this.yScale_ph( phase(d[1], d[2]) ); });

    this.plot_ph.append("path")
        .datum(data)
        .attr("id", id)
        .attr("name", name)
        .attr("d", line_ph);

    var line_mag = d3.svg.line()
        .x(function(d) { return _this.xScale_mag( d[0] ); })
        .y(function(d) { return _this.yScale_mag( mag(d[1], d[2]) ); });

    this.plot_mag.append("path")
        .datum(data)
        .attr("id", id)
        .attr("d", line_mag);
};
