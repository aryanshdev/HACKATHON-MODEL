import pandas as pd
import joblib
import numpy as np

# Load the saved label encoder
label_encoder = joblib.load('label_encoder.pkl')
print("Label encoder loaded from 'label_encoder.pkl'.")

# Load the saved Random Forest model
rf_clf = joblib.load('random_forest_model.pkl')
print("Random Forest model loaded from 'random_forest_model.pkl'.")

# Load the new data
new_data = pd.read_csv("HACKATHON-MODEL/fetal_health_modified.csv")

# Load the original training feature columns
X_train = pd.read_csv("X_train.csv")

# Ensure new data contains the same features as the training data
required_features = X_train.columns

# Remove any target column if present
required_features = [col for col in required_features if col != 'fetal_health']

# Check if all required features are present in the new data
missing_features = [feature for feature in required_features if feature not in new_data.columns]
if missing_features:
    raise ValueError(f"The following required features are missing from the new data: {missing_features}")

# Ensure new_data only contains the required features
X_new = new_data[required_features]

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
original_predictions = label_encoder.inverse_transform(predictions)

# Add predictions to the new data for reference
new_data['predicted_fetal_health'] = predictions
new_data['original_predicted_fetal_health'] = original_predictions

# Print the DataFrame with predictions
print(new_data.head())

# Save the DataFrame with predictions
new_data.to_csv("HACKATHON-MODEL/fetal_health_predictions.csv", index=False)
print("Predictions saved to 'HACKATHON-MODEL/fetal_health_predictions.csv'.")
