import pandas as pd
from sklearn import preprocessing 
import joblib



data = pd.read_csv("HACKATHON-MODEL/fetal_health.csv")


def replace_values(data, column, old_value, new_value):
    try:
        data[column] = data[column].replace(old_value, new_value)
    except Exception as e:
        print(f"Error occurred while replacing values in {column}: ", str(e))
    return data

# Replace 'normal' with 1 and 'pathological' with 2 in the 'target

data = replace_values(data, 'fetal_health', 1, "ok")
data = replace_values(data, 'fetal_health', 2, "good")
data = replace_values(data, 'fetal_health', 3, "very good")

print(data.head())

#data.to_csv("HACKATHON-MODEL/fetal_health_modified.csv", index=False)
#print("Modified data saved to HACKATHON-MODEL/fetal_health_modified.csv")


# Import dataset 
df = pd.read_csv('HACKATHON-MODEL/fetal_health_modified.csv') 
  
print(df['fetal_health'].unique())

label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
df['fetal_health']= label_encoder.fit_transform(df['fetal_health']) 
  
print(df['fetal_health'].unique())



