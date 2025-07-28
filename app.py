from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
# app.py
print("Python environment is working correctly.")

# Load your trained model (replace with your actual model file)
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request (adjust based on your features)
        data = request.json
        soil_type = data.get('ph')
        weather = data.get('rainfall')
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))

        # Prepare data for prediction (this depends on how your model is trained)
        input_data = np.array([[soil_type, weather, temperature, humidity]])

        # Make prediction
        prediction = model.predict(input_data)

        # Return prediction as JSON
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
