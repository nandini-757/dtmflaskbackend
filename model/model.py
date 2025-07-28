import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import pickle
# Load dataset
data = pd.read_csv('model\CropData.csv')

# Check for missing values
data = data.dropna()  # or handle missing values as required

# Label Encoding for Crop column
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Feature Scaling
scaler = StandardScaler()
features = ['temperature', 'rainfall', 'ph', 'humidity']
data[features] = scaler.fit_transform(data[features])

# Splitting data into training and testing sets
X = data[features]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize Random Forest model
model = RandomForestClassifier(n_estimators=100)
# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(X_train.shape)  # Should be (n_samples, 4)
print(X_test.shape)   # Should be (n_samples, 4)
print(X_test)
# Create a SHAP explainer
explainer = shap.TreeExplainer(model)

# Get SHAP values for the test set
choosen_instance = X_test.loc[[71]]
shap_values = explainer.shap_values(choosen_instance)
print(shap_values)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0],choosen_instance)
# print(f"SHAP values shape: {shap_values[0].shape}") 
# #print(shap_values)


# feature_names = X_test.columns  # X_test should be your DataFrame with the input features

# # Print the feature names
# print("Feature names:", feature_names.tolist())





# shap.initjs()
# instance_index = 0  # or any valid index in X_test
# shap.force_plot(explainer.expected_value[0], shap_values[instance_index], X_test.iloc[instance_index])
# #shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])


with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training complete and saved as model.pkl")
