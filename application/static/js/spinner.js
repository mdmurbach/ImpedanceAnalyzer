$('#loadingDiv').hide();
$(document).ajaxStart(function() {
    $("#loadingDiv").show();
});

$(document).ajaxStop(function() {
    $("#loadingDiv").hide();
});
