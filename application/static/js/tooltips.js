/* Add tooltips here... */
$('#info-fileupload').prop('title', 'The uploaded file should be a comma-separated file with the first column as frequency, second column as real impedance, third column as imaginary impedance.')

$('#info-parameters').prop('title', 'The error is given as the estimated one-sigma standard deviation');

/* enables tooltips to be shown */
$(function () {
    $('[data-toggle="tooltip"]').tooltip({ placement: 'bottom'})
})
