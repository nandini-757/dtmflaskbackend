from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lime import lime_tabular
from googletrans import Translator
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Load the dataset
df = pd.read_csv('model/CropData.csv')

# Drop 's.no' column
df = df.drop(columns=['sno'])

# Encode crop labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Feature scaling
scaler = StandardScaler()
features = ['temperature', 'rainfall', 'ph', 'humidity']
df[features] = scaler.fit_transform(df[features])

# Splitting data into training and testing sets
X = df[features]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost Classifier
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Initialize LIME explainer
explainer = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=label_encoder.classes_, verbose=True, mode='classification')

def generate_natural_language_explanation(prediction, contributions, crop_name):
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
            negative_contributions += abs(weight)  # Using absolute value to count negative impact
            if feature == "rainfall":
                explanation += f"The {feature} is low, which slightly discourages this crop.\n"
            elif feature == "humidity":
                explanation += f"High {feature} discourages this crop a little.\n"
            else:
                explanation += f"{feature.capitalize()} discourages this crop slightly.\n"

    # Conditional recommendation based on contributions
    if positive_contributions > negative_contributions:
        explanation += f"Overall, we have more positives, so planting {crop_name} seems like a good choice based on the favorable conditions."
    else:
        explanation += f"Overall, we have more negatives, so planting {crop_name} might not be advisable due to the unfavorable conditions."

    return explanation

def translate_explanation(explanation, dest_language):
    translator = Translator()
    translation = translator.translate(explanation, dest=dest_language)
    return translation.text

# Flask API
# app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    temperature = float(data['temperature'])
    rainfall = float(data['rainfall'])
    ph = float(data['ph'])
    humidity = float(data['humidity'])
    language = data.get('language', 'en')  # Default language is English

    # Scale the user input features
    user_input = pd.DataFrame([[temperature, rainfall, ph, humidity]], columns=features)
    user_input_scaled = scaler.transform(user_input)

    # Make a prediction
    predicted_label = model.predict(user_input_scaled)[0]
    predicted_crop = label_encoder.inverse_transform([predicted_label])[0]  # Get the actual crop name

    # Generate LIME explanation for the user's input
    lime_explanation = explainer.explain_instance(user_input_scaled[0], model.predict_proba)
    
    # Translate LIME explanation into natural language
    contributions = {feature: weight for feature, weight in lime_explanation.as_list()}
    explanation = generate_natural_language_explanation(predicted_label, contributions, predicted_crop)

    # Translate explanation if requested
    translated_explanation = translate_explanation(explanation, language) if language != 'en' else explanation

    # Return the prediction and explanation
    return jsonify({
        'predicted_crop': predicted_crop,
        'explanation': translated_explanation
    })

if __name__ == '__main__':
    app.run(debug=True)
