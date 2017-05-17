function show_download() {
    $('#btn-download').removeClass('hidden')
}

function download_csv() {
    console.log("Downloading!");

    var csv = 'data:text/csv;charset=utf-8,'

    results.forEach(function(result) {
        csv += result.model + "\n"

        csv += "Parameter,Units,Value,Error Estimate\n"

        for (p = 0; p < result.names.length; p++) {
            row = result.names[p] + ',' + result.units[p] + ',' +
                  result.values[p] + ',' + result.errors[p] + "\n"
            csv += row
        }
        csv += "\n"
        
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
