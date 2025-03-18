document.addEventListener('DOMContentLoaded', () => {
    // Update slider values in real-time
    const sliders = ['rain', 'ffmc', 'dmc', 'isi'];
    sliders.forEach(id => {
        const slider = document.getElementById(id);
        const valueSpan = document.getElementById(`${id}-value`);
        slider.oninput = () => valueSpan.textContent = slider.value;
    });

    // Handle form submission
    const form = document.getElementById('prediction-form');
    form.onsubmit = async (e) => {
        e.preventDefault();

        // Get form data
        const formData = new FormData(form);

        // Send POST request to /predict
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            // Display result
            const resultDiv = document.getElementById('result');
            const fireStatus = document.getElementById('fire-status');
            const tipsDiv = document.getElementById('tips');
            const chanceValue = document.getElementById('chance-value');
            const progressFill = document.getElementById('progress-fill');

            resultDiv.style.display = 'block';
            if (result.fire_detected) {
                fireStatus.textContent = '🔥 Fire Detected!';
                fireStatus.style.color = 'red';
                tipsDiv.style.display = 'block';
            } else {
                fireStatus.textContent = '✅ No Fire Risk';
                fireStatus.style.color = 'green';
                tipsDiv.style.display = 'none';
            }

            chanceValue.textContent = result.chance_of_fire;
            progressFill.style.width = `${result.chance_of_fire}%`;
            progressFill.className = 'progress-fill ' + result.progress_color;
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while predicting. Please try again.');
        }
    };
});