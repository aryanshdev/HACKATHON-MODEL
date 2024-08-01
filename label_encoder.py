import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def encode_target_column(df, target_column):
    """
    This function checks if the target column of the DataFrame is of type object.
    If it is, it initializes and runs a Label Encoder on it.
    If not, it does nothing.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    target_column (str): The name of the target column.

    Returns:
    pd.DataFrame: The DataFrame with the encoded target column if it was of type object.
    LabelEncoder: The label encoder used for encoding (if applicable).
    """
    if df[target_column].dtype == 'object':
        label_encoder = preprocessing.LabelEncoder()

        # Encode labels in 'fetal_health' column
        df['target'] = label_encoder.fit_transform(df['target'])

        print("Unique values after encoding:", df['target'].unique())
        print(df)
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        return df, label_encoder
    
    else:
        print(f"Target column '{target_column}' is not of type object. No encoding applied.")
        return df, None

# Example usage
if __name__ == "__main__":
    
    data = pd.read_csv("filename")
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    df, encoder = encode_target_column(df, 'target')
    
    print("Modified DataFrame:")
    print(df)
