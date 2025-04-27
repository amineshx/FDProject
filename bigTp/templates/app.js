document.getElementById('scatter-form').addEventListener('submit', function(event) {
    event.preventDefault(); 

    const formData = new FormData(this);

    fetch('{{ url_for("generate_scatter") }}', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        document.getElementById('scatter-plot-section').innerHTML = html;
    })
    .catch(error => console.error('Error:', error));
});