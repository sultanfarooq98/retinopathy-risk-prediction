from flask import Flask, request, render_template
import numpy as np
import pickle
from models import load_models

# Initialize the Flask app
app = Flask(__name__)

# Load models and scaler
log_reg, rf, svm, scaler = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract the feature values (excluding the model choice)
        feature_values = [float(x) for key, x in request.form.items() if key != 'model_choice']

        # Convert the feature values into a NumPy array
        features_array = np.array([feature_values])

        # Scale the features before prediction
        scaled_features = scaler.transform(features_array)

        # Get the selected model
        model_choice = request.form.get('model_choice')

        if model_choice == 'log_reg':
            prediction = log_reg.predict(scaled_features)
        elif model_choice == 'rf':
            prediction = rf.predict(scaled_features)
        elif model_choice == 'svm':
            prediction = svm.predict(scaled_features)
        else:
            prediction = "Invalid model selected"

        # Return the prediction to the webpage
        return render_template('index.html', prediction_text=f'Predicted Retinopathy Level: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
