import pandas as pd
from sklearn import preprocessing 
import joblib

# Load the initial data
data = pd.read_csv('HACKATHON-MODEL/fetal_health.csv')

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

print("Unique values after encoding:", df['fetal_health'].unique())

print(df)

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Save the label encoder to a file
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Label encoder saved as 'label_encoder.pkl'.")

# Print the label mapping
print("Label mapping (original: encoded):", label_mapping)
