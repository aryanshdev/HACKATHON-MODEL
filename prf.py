import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the label encoder and Random Forest model
label_encoder = joblib.load('label_encoder.pkl')
print("Label encoder loaded from 'label_encoder.pkl'.")

rf_clf = joblib.load('random_forest_model.pkl')
print("Random Forest model loaded from 'random_forest_model.pkl'.")

# Load the new data
new_data = pd.read_csv("HACKATHON-MODEL/fetal_health_modified.csv")

# Load training data to get the required features
X_train = pd.read_csv("X_train.csv")
required_features = X_train.columns

# Remove any target column if present
required_features = [col for col in required_features if col != 'fetal_health']

# Check if all required features are present in the new data
missing_features = [feature for feature in required_features if feature not in new_data.columns]
if missing_features:
    raise ValueError(f"The following required features are missing from the new data: {missing_features}")

# Ensure new_data only contains the required features
X_new = new_data[required_features]

# Assume the new data contains the true labels for evaluation
y_true = new_data['fetal_health'] if 'fetal_health' in new_data.columns else None

# Make predictions using the loaded Random Forest model
predictions = rf_clf.predict(X_new)

# Check for unseen labels in predictions
seen_labels = set(label_encoder.transform(label_encoder.classes_))
unseen_labels = set(predictions) - seen_labels

if unseen_labels:
    print(f"Unseen labels in predictions: {unseen_labels}")
    
    # Find the most common class in the training data
    most_common_class = pd.Series(label_encoder.transform(label_encoder.classes_)).mode()[0]
    
    # Map unseen labels to the most common class
    predictions = np.where(np.isin(predictions, list(unseen_labels)), most_common_class, predictions)

# Inverse transform the predictions to get original labels
original_predictions = label_encoder.inverse_transform(predictions.astype(int))

# Add predictions to the new data for reference
new_data['predicted_fetal_health'] = predictions
new_data['original_predicted_fetal_health'] = original_predictions

# Print the DataFrame with predictions
print(new_data.head())

# Save the DataFrame with predictions
new_data.to_csv("HACKATHON-MODEL/fetal_health_predictions333.csv", index=False)
print("Predictions saved to 'HACKATHON-MODEL/fetal_health_predictions33.csv'.")

# If true labels are available, evaluate the model's performance
if y_true is not None:
    y_true_encoded = label_encoder.transform(y_true)
    accuracy = accuracy_score(y_true_encoded, predictions)
    report = classification_report(y_true_encoded, predictions)
    matrix = confusion_matrix(y_true_encoded, predictions)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", matrix)
