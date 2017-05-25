var Nyquist = function(config) {

    this.outerWidth = config.outerWidth;
    this.outerHeight = config.outerHeight;
    this.element = config.element;
    this.data = config.data;

    if(this.outerWidth < this.outerHeight) {
        this.outerHeight = this.outerWidth;
    } else {
        this.outerWidth = this.outerHeight;
    }
    
    this.margin = {top: 20, right: 20, bottom: 80, left: 80};
    this.width = this.outerWidth - this.margin.left - this.margin.right;
    this.height = this.outerHeight - this.margin.top - this.margin.bottom;


    this.draw();
}

Nyquist.prototype.draw = function() {

    var svg = d3.select(this.element).append('svg')
        .attr('width', this.outerWidth)
        .attr('height', this.outerHeight)

    this.plot = svg.append('g')
        .attr('transform', 'translate(' + this.margin.left + "," + this.margin.top + ')')

    this.createScales();
    this.addAxes();
    this.initializeData();
}

Nyquist.prototype.createScales = function(){

    var max = d3.max( [d3.max(this.data, function(d, i) { return d[1]} ) - d3.min(this.data, function(d, i) { return d[1]} ),
                            d3.max(this.data, function(d, i) { return d[2]} ) - d3.min(this.data, function(d, i) { return d[2]} )]);

    this.xScale = d3.scale.linear()
        .range([0, this.width])
        .domain([ d3.min(this.data, function(d, i) { return d[1]} ) - max*.1,  d3.min(this.data, function(d, i) { return d[1]} ) + max*1.2]);

    this.yScale = d3.scale.linear()
        .range([this.height, 0])
        .domain([0, max*1.3]);

}

Nyquist.prototype.addAxes = function() {

    var xAxis = d3.svg.axis()
        .scale(this.xScale)
        .ticks(5)
        .orient('bottom');

    var yAxis = d3.svg.axis()
        .scale(this.yScale)
        .ticks(5)
        .orient('left');

    this.plot.append('g')
        .attr('class', 'x axis')
        .attr('transform', 'translate(0,' + this.height + ')')
        .call(xAxis)

    this.plot.append('g')
        .attr('class', 'y axis')
        .attr('transform', 'translate(0,' + 0 + ')')
        .call(yAxis)

    svgWidth = d3.select(this.element).select('svg')
                    .node().getAttribute('width')
    svgHeight = d3.select(this.element).select('svg')
                    .node().getAttribute('height')

    var ylabel = d3.select(this.element).append("div.label")

    ylabel.style("transform", "rotate(-90deg)")
        .style("position", "absolute")
        .style("left", -.2*this.margin.left + "px")
        .style("top", svgHeight/2 + "px")
        .style("text-anchor", "right")

    var xlabel = d3.select(this.element).append("div.label")

    xlabel.style("position", "absolute")
        .style("left", svgWidth/2 + "px")
        .style("top", svgHeight - .6*this.margin.bottom + "px")
        .style("text-anchor", "middle")

    katex.render("-Z_{1,1}^{\\prime\\prime} \\mathrm{ [Ohms]}", ylabel.node())

    katex.render("Z_{1,1}^{\\prime} \\mathrm{ [Ohms]}", xlabel.node())
}


Nyquist.prototype.initializeData = function(){
    var _this = this;

    var circles = this.plot.selectAll("circle").data(this.data);

    circles.enter().append('circle');
    circles
        .attr("class", "default-circle")
        .attr("cx", function (d) { return _this.xScale( d[1] ); } )
        .attr("cy", function (d) { return _this.yScale( -d[2] ); } )
        .attr("r", 5)
        .on("mouseover", function(d,i) {
            d3.select(this)
                .classed("selected-circle", true)
                .attr("r", 10);

            d3.select("#phase").selectAll("circle")
                .data(_this.data)
                .classed("selected-circle", function(d2, i2) {return i2 == i; })
                .attr("r", function(d2, i2) { if(i2 == i) { return 10; } else { return 5;} });

            d3.select("#magnitude").selectAll("circle")
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

            d3.select("#magnitude").selectAll("circle")
                .data(_this.data)
                .attr("class", "default-circle")
                .attr("r", 5);
        });

}

Nyquist.prototype.addPoints = function (data, class_name, id) {

    var _this = this;

    var circles = this.plot.selectAll("circle#" + id).data(data);

    circles.enter().append('circle');
    circles
        .attr("id", id)
        .attr("cx", function (d) { return _this.xScale( d[1] ); } )
        .attr("cy", function (d) { return _this.yScale( -d[2] ); } )
        .attr("r", 2)
};

Nyquist.prototype.addModel = function (data, class_name, id, color /* optional */) {

    var _this = this;

    var line = d3.svg.line()
        .x(function(d) { return _this.xScale( d[1] ); })
        .y(function(d) { return _this.yScale( -d[2] ); });

    this.plot.append("path")
      .datum(data)
      .attr("class", class_name)
      .style("stroke", color)
      .attr("id", id)
      .attr("d", line);
};

Nyquist.prototype.updateModel = function (data, scale, class_name, id) {

    var _this = this;

    impedances = this.plot.selectAll("circle." + class_name).data(data)

    // define the line
    var line = d3.svg.line()
        .x(function(d) { return _this.xScale(d.real/scale); })
        .y(function(d) { return _this.yScale(d.imag/scale); });

    d3.selectAll('.' + id).remove();

    this.plot.append("path")
        .datum(data)
        .attr("id", id)
        .attr("class", class_name)
        .transition().duration(50)
        .attr("d", line);

    impedances
        .enter().append("circle")
        .attr("class", class_name)
        .attr("id", id)
        .attr("r", 5)

    impedances
        .transition().duration(50)
        .attr("cx", function(d) {return _this.xScale(d.real/scale)})
        .attr("cy", function(d) { return _this.yScale(d.imag/scale)})

    impedances
        .exit().remove();

};

Nyquist.prototype.addLines = function (data, class_name, id) {

    var _this = this;

    // define the line
    var line = d3.svg.line()
        .x(function(d) { return _this.xScale(d.real); })
        .y(function(d) { return _this.yScale(d.imag); });

    this.plot.append("path")
        .datum(data)
        .attr("class", class_name)
        .attr("id", id)
        .attr("d", line);

};

Nyquist.prototype.clear = function (descriptor) {

    d3.select(this.element).selectAll(descriptor).remove();

};
