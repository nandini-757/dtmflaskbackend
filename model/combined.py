from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lime import lime_tabular
from googletrans import Translator
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load and preprocess datasets
df_crop_recommendation = pd.read_csv('model/CropData.csv')
df_irrigation_recommendation = pd.read_csv('model/prodata.csv')

# Crop Recommendation Model Setup
df_crop_recommendation = df_crop_recommendation.drop(columns=['sno'])
crop_label_encoder = LabelEncoder()
df_crop_recommendation['label'] = crop_label_encoder.fit_transform(df_crop_recommendation['label'])
crop_scaler = StandardScaler()
crop_features = ['temperature', 'rainfall', 'ph', 'humidity']
df_crop_recommendation[crop_features] = crop_scaler.fit_transform(df_crop_recommendation[crop_features])
X_crop = df_crop_recommendation[crop_features]
y_crop = df_crop_recommendation['label']
X_crop_train, X_crop_test, y_crop_train, y_crop_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)
crop_recommendation_model = xgb.XGBClassifier(random_state=42)
crop_recommendation_model.fit(X_crop_train, y_crop_train)

# Irrigation Recommendation Model Setup
irrigation_features = ['CropDays', 'Soil Moisture', 'Soil Temperature', 'Temperature', 'Humidity']
X_irrigation = df_irrigation_recommendation[irrigation_features]
y_irrigation = df_irrigation_recommendation['Irrigation(Y/N)']
X_irrig_train, X_irrig_test, y_irrig_train, y_irrig_test = train_test_split(X_irrigation, y_irrigation, test_size=0.2, random_state=42)
irrigation_scaler = StandardScaler()
X_irrig_train = irrigation_scaler.fit_transform(X_irrig_train)
X_irrig_test = irrigation_scaler.transform(X_irrig_test)
irrigation_recommendation_model = RandomForestClassifier(n_estimators=100, random_state=42)
irrigation_recommendation_model.fit(X_irrig_train, y_irrig_train)
joblib.dump(irrigation_recommendation_model, 'irrigation_model.pkl')

# Initialize LIME explainers
crop_lime_explainer = lime_tabular.LimeTabularExplainer(X_crop_train.values, feature_names=crop_features, class_names=crop_label_encoder.classes_, verbose=True, mode='classification')
irrigation_lime_explainer = lime_tabular.LimeTabularExplainer(X_irrig_train, feature_names=irrigation_features, class_names=['No', 'Yes'], verbose=True, mode='classification')

# Custom explanation functions
def generate_crop_natural_language_explanation(prediction, contributions, crop_name):
    explanation = f"The model suggests planting {crop_name}.\n"
    positive_contributions = 0
    negative_contributions = 0

    for feature, weight in contributions.items():
        if weight > 0:
            positive_contributions += weight
            if feature == "rainfall":
                explanation += f"The {feature} is favorable for this crop.\n"
            elif feature == "temperature":
                explanation += f"The {feature} is favorable for this crop.\n"
            elif feature == "ph":
                explanation += f"The {feature} level is within a range that supports planting {crop_name}.\n"
            else:
                explanation += f"{feature.capitalize()} supports the growth of this crop.\n"
        else:
            negative_contributions += abs(weight)
            if feature == "rainfall":
                explanation += f"The {feature} is low, which slightly discourages this crop.\n"
            elif feature == "humidity":
                explanation += f"High {feature} discourages this crop a little.\n"
            else:
                explanation += f"{feature.capitalize()} discourages this crop slightly.\n"

    if positive_contributions > negative_contributions:
        explanation += f"Overall, we have more positives, so planting {crop_name} seems like a good choice based on the favorable conditions."
    else:
        explanation += f"Overall, we have more negatives, so planting {crop_name} might not be advisable due to the unfavorable conditions."

    return explanation

def generate_irrigation_natural_language_explanation(prediction, contributions):
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
    
    if positive_contributions > negative_contributions:
        explanation += "Overall, the conditions are favorable for irrigation."
    else:
        explanation += "Overall, conditions suggest no irrigation might be needed."

    return explanation

def translate_explanation(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    data = request.json
    temperature = float(data['temperature'])
    rainfall = float(data['rainfall'])
    ph = float(data['ph'])
    humidity = float(data['humidity'])
    language = data.get('language', 'en')
    crop_input = pd.DataFrame([[temperature, rainfall, ph, humidity]], columns=crop_features)
    crop_input_scaled = crop_scaler.transform(crop_input)
    predicted_crop_label = crop_recommendation_model.predict(crop_input_scaled)[0]
    predicted_crop_name = crop_label_encoder.inverse_transform([predicted_crop_label])[0]
    crop_explanation_instance = crop_lime_explainer.explain_instance(crop_input_scaled[0], crop_recommendation_model.predict_proba)
    crop_contributions = {feature: weight for feature, weight in crop_explanation_instance.as_list()}
    explanation_text = generate_crop_natural_language_explanation(predicted_crop_label, crop_contributions, predicted_crop_name)
    translated_explanation = translate_explanation(explanation_text, language) if language != 'en' else explanation_text
    return jsonify({'predicted_crop': predicted_crop_name, 'explanation': translated_explanation})

@app.route('/predict_irrigation', methods=['POST'])
def predict_irrigation():
    data = request.json
    crop_days = float(data['cropDays'])
    soil_moisture = float(data['soilMoisture'])
    soil_temp = float(data['soilTemp'])
    temperature = float(data['irrigationTemp'])
    humidity = float(data['irrigationHumidity'])
    language = data.get('language', 'en')
    irrigation_input = pd.DataFrame([[crop_days, soil_moisture, soil_temp, temperature, humidity]], columns=irrigation_features)
    irrigation_input_scaled = irrigation_scaler.transform(irrigation_input)
    irrigation_model = joblib.load('irrigation_model.pkl')
    irrigation_pred_label = irrigation_model.predict(irrigation_input_scaled)[0]
    irrigation_prediction = 'Yes' if irrigation_pred_label == 1 else 'No'
    irrigation_explanation_instance = irrigation_lime_explainer.explain_instance(irrigation_input_scaled[0], irrigation_model.predict_proba)
    irrigation_contributions = {feature: weight for feature, weight in irrigation_explanation_instance.as_list()}
    explanation_text = generate_irrigation_natural_language_explanation(irrigation_prediction, irrigation_contributions)
    translated_explanation = translate_explanation(explanation_text, language) if language != 'en' else explanation_text
    return jsonify({'predicted_irrigation': irrigation_prediction, 'explanation': translated_explanation})

if __name__ == '__main__':
    app.run(debug=True)
