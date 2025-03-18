from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the models
model = joblib.load("Classification1.joblib")
model2 = joblib.load("Regression.joblib")

@app.route('/')
def landing():
    return render_template('Home.html')
@app.route('/Predict.html')
def predict_page():
    return render_template('Predict.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        rain = float(request.form.get('rain', 0.0))
        ffmc = float(request.form.get('ffmc', 0.0))
        dmc = float(request.form.get('dmc', 0.0))
        isi = float(request.form.get('isi', 0.0))

        # Prepare input data
        input_data = np.array([[rain, ffmc, dmc, isi]])

        # Make predictions
        prediction = model.predict(input_data)[0]
        prediction2 = model2.predict(input_data)[0]
        value=prediction2*3.5

        # Scale chance of fire (adjust based on your model's output range)
        chance_of_fire = min(max(value , 0), 100)  # Example: scale 0-5 to 0-100

        # Prepare response
        result = {
            'fire_detected': bool(prediction == 1 and value>=50),
            'chance_of_fire': round(chance_of_fire, 2),
            'progress_color': 'green' if chance_of_fire < 50 else 'red'
        }
        return jsonify(result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)