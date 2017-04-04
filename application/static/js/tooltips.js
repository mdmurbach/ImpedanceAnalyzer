/* Add tooltips to be show here... */
$('#info-fileupload').prop('title', 'TODO: The uploaded .csv file will be interpreted with the first column as frequency, second column as real impedance, third column as imaginary impedance.')
$('#info-parameters').prop('title', 'Error indicates estimated standard deviation of fit parameters');

/* enables tooltips to be shown */
$(function () {
    $('[data-toggle="tooltip"]').tooltip({ placement: 'bottom'})
})
