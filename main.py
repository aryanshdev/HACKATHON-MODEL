import pandas as pd
from sklearn import preprocessing 
import joblib

# Load the initial data
data = pd.read_csv("HACKATHON-MODEL/fetal_health.csv")

# Function to replace values in a specified column
def replace_values(data, column, old_value, new_value):
    try:
        data[column] = data[column].replace(old_value, new_value)
    except Exception as e:
        print(f"Error occurred while replacing values in {column}: ", str(e))
    return data

# Replace values in 'fetal_health' column
data = replace_values(data, 'fetal_health', 1, "ok")
data = replace_values(data, 'fetal_health', 2, "good")
data = replace_values(data, 'fetal_health', 3, "very good")

# Save the modified data to a new CSV file
data.to_csv("HACKATHON-MODEL/fetal_health_modified.csv", index=False)
print("Modified data saved to 'HACKATHON-MODEL/fetal_health_modified.csv'")

# Load the modified dataset
df = pd.read_csv('HACKATHON-MODEL/fetal_health_modified.csv')

# Print unique values in 'fetal_health' column before encoding
print("Unique values before encoding:", df['fetal_health'].unique())

# Initialize the label encoder
label_encoder = preprocessing.LabelEncoder()

# Encode labels in 'fetal_health' column
df['fetal_health'] = label_encoder.fit_transform(df['fetal_health'])

# Print unique values in 'fetal_health' column after encoding
print("Unique values after encoding:", df['fetal_health'].unique())

# Print the modified DataFrame
print(df)

# Create a dictionary for the label mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Save the label encoder to a file
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Label encoder saved as 'label_encoder.pkl'.")

# Print the label mapping
print("Label mapping (original: encoded):", label_mapping)
