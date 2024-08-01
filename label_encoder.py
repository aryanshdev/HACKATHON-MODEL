import pandas as pd
from sklearn import preprocessing
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def encode_target_column(df, target_column):

    if df[target_column].dtype == 'object':
        label_encoder = preprocessing.LabelEncoder()

        df['fetal_health'] = label_encoder.fit_transform(df['fetal_health'])

        print("Unique values after encoding:", df['fetal_health'].unique())
        print(df)
        label_mapping = dict(zip (label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
     
        # Save the label encoder to a file
        joblib.dump(label_encoder, 'label_encoder.pkl')
        print("Label encoder saved as 'label_encoder.pkl'.")
        # Print the label mapping
        print("Label mapping (original: encoded):", label_mapping)


        return df, label_encoder
    
    else:
        print(f"Target column '{target_column}' is not of type object. No encoding applied.")
        return df, None

# Example usage
if __name__ == "__main__":
    
    data = pd.read_csv("HACKATHON-MODEL/fetal_health.csv")
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    df, encoder = encode_target_column(df, 'fetal_health')
    
    print("Modified DataFrame:")
    print(df)
