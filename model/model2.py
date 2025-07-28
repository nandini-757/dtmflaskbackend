from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from lime import lime_tabular
from sklearn.metrics import accuracy_score, classification_report
from flask_cors import CORS
import joblib
from googletrans import Translator

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load and preprocess the dataset
data = pd.read_csv('model/prodata.csv')
features = ['CropDays', 'Soil Moisture', 'Soil Temperature', 'Temperature', 'Humidity']
X = data[features]
y = data['Irrigation(Y/N)']

# Train-test split and feature scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'precision_irrigation_model.pkl')

# Initialize LIME explainer
explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=features, class_names=['No', 'Yes'], verbose=True, mode='classification')

# Function to generate a natural language explanation
def generate_natural_language_explanation(prediction, contributions):
    explanation = f"The model recommends {'irrigation' if prediction == 'Yes' else 'no irrigation'}.\n"
    positive_contributions = 0
    negative_contributions = 0
    
    for feature, weight in contributions.items():
        if weight > 0:
            positive_contributions += weight
            explanation += f"{feature.capitalize()} positively supports irrigation due to its value.\n"
        else:
            negative_contributions += abs(weight)
            explanation += f"{feature.capitalize()} suggests no irrigation due to its influence.\n"
    
    # Summarize based on contributions
    if positive_contributions > negative_contributions:
        explanation += "Overall, the conditions are favorable for irrigation."
    else:
        explanation += "Overall, conditions suggest no irrigation might be needed."

    return explanation

# Function for translating the explanation if needed
def translate_explanation(explanation, dest_language):
    translator = Translator()
    translation = translator.translate(explanation, dest=dest_language)
    return translation.text

# Define the prediction and explanation endpoint
@app.route('/predict_irrigation', methods=['POST'])
def predict_irrigation():
    data = request.json
    crop_days = float(data['CropDays'])
    soil_moisture = float(data['Soil Moisture'])
    soil_temp = float(data['Soil Temperature'])
    temperature = float(data['Temperature'])
    humidity = float(data['Humidity'])
    language = data.get('language', 'en')  # Default language is English

    # Scale the user input
    user_input = pd.DataFrame([[crop_days, soil_moisture, soil_temp, temperature, humidity]], columns=features)
    user_input_scaled = scaler.transform(user_input)

    # Load the model and make a prediction
    model = joblib.load('precision_irrigation_model.pkl')
    predicted_label = model.predict(user_input_scaled)[0]
    prediction = 'Yes' if predicted_label == 1 else 'No'

    # Generate a LIME explanation
    lime_explanation = explainer.explain_instance(user_input_scaled[0], model.predict_proba)
    contributions = {feature: weight for feature, weight in lime_explanation.as_list()}

    # Create natural language explanation
    explanation = generate_natural_language_explanation(prediction, contributions)

    # Translate if needed
    translated_explanation = translate_explanation(explanation, language) if language != 'en' else explanation

    # Return the prediction and explanation
    return jsonify({
        'predicted_irrigation': prediction,
        'explanation': translated_explanation
    })

if __name__ == '__main__':
    app.run(debug=True)
