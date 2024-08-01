import numpy as np
import pandas as pd
import os


# Call to createdataframe
def createDataFrame(path):

    file_extension = os.path.splitext(path)[1]

    if(file_extension==".csv"):
        df =pd.read_csv(path)

    elif file_extension in [".xlsx",".xls"]:
        df = pd.read_excel(path)

    else:
        return "Unsupported File extension, please input csv, xlsx or xls file type."

    return df

# Can only return the 1st valid file it encounters.
def find_file_in_directory(directory, extensions= ['.csv', '.xlsx', '.xls']):
    # List all files in the directory
    for file_name in os.listdir(directory):
        # Check if the file has the desired extension
        if any(file_name.endswith(ext) for ext in extensions):
            return os.path.join(directory, file_name)
    return None


def clean_column_names(df):
    # Original column names
    original_columns = df.columns.tolist()
    # Clean column names
    df.columns = df.columns.str.strip()  # Remove leading and trailing whitespace
    df.columns = df.columns.str.lower()  # Convert to lowercase
    df.columns = df.columns.str.replace(r'[^a-z0-9]', '_', regex=True)  # Replace non-alphanumeric characters with underscores
    
     # Create description of changes
    cleaned_columns = df.columns.tolist()
    changes = []
    
    for orig, cleaned in zip(original_columns, cleaned_columns):
        if orig != cleaned:
            changes.append(f"'{orig}' -> '{cleaned}'")
    
    if changes:
        changes_description = "Column name changes:\n" + "\n".join(changes)
    else:
        changes_description = "No changes to column names."
    
    return df, changes_description


# Takes in the dataframe and a String of the columns to be dropped; seperated by comma.
def drop_columns_from_string(df, column_string):
    # Split the string by commas and trim whitespace
    columns_to_drop = [col.strip() for col in column_string.split(',')]
    
    # Check for invalid column names
    invalid_columns = [col for col in columns_to_drop if col not in df.columns]
    
    if invalid_columns:
        # If there are invalid columns, return a message and do not modify the DataFrame
        invalid_message = f"Invalid column names: {', '.join(invalid_columns)}"
        return df, invalid_message
    
    # Drop the specified columns from the DataFrame
    df_dropped = df.drop(columns=[col for col in columns_to_drop], errors='ignore')
    
    # Provide feedback on which columns were dropped
    dropped_columns = [col for col in columns_to_drop if col in df.columns]
    if dropped_columns:
        message = f"Dropped columns: {', '.join(dropped_columns)}"
    else:
        message = "No columns were dropped."
    
    return df_dropped, message

# Removes duplicate rows and returns message as such.
def remove_duplicates(df):
    # Check for duplicate rows
    if df.duplicated().any():
        # Remove duplicate rows
        df_cleaned = df.drop_duplicates()
        # Provide feedback on the number of duplicates removed
        num_duplicates = df.shape[0] - df_cleaned.shape[0]
        message = f"Removed {num_duplicates} duplicate row(s)."
    else:
        # No duplicates found
        df_cleaned = df
        message = "No duplicate rows found."
    
    return df_cleaned, message

# Handle missing values 
def handle_missing_values(df):
    
    # Copy the DataFrame to avoid modifying the original data
    df_processed = df.copy()
    
    # Iterate over each column in the DataFrame
    for column in df_processed.columns:
        # Calculate the percentage of missing values
        missing_percentage = df_processed[column].isna().mean() * 100
        
        if missing_percentage > 55:
            # Drop the column if missing value percentage is greater than 55%
            df_processed.drop(columns=[column], inplace=True)
            message = f"Dropped column '{column}' due to missing value percentage of {missing_percentage:.2f}%."
        else:
            # Fill missing values with the mean of the column
            mean_value = df_processed[column].mean()
            df_processed[column].fillna(mean_value, inplace=True)
            message = f"Filled missing values in column '{column}' with mean value {mean_value:.2f}."

    
    return df_processed,message

# Convert to Numeric
def convert_to_numeric(df, columns):
    """
    Convert specified columns in the DataFrame to numeric values (int or float).
    
    Parameters:
    df (pd.DataFrame): The input DataFrame with columns to convert.
    columns (str): Comma-separated string of column names to convert.
    
    Returns:
    pd.DataFrame: DataFrame with specified columns converted to numeric.
    str: Message indicating the result of the conversion process.
    """
    # Split the comma-separated string into a list of columns
    column_list = [col.strip() for col in columns.split(',')]
    
    message = []
    
    # Copy the DataFrame to avoid modifying the original data
    df_converted = df.copy()
    
    for column in column_list:
        if column in df_converted.columns:
            try:
                # Attempt to convert the column to numeric
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
                message.append(f"Column '{column}' converted to numeric.")
            except Exception as e:
                message.append(f"Error converting column '{column}': {e}")
        else:
            message.append(f"Column '{column}' not found in DataFrame.")
    
    # Check if there were any issues with conversion
    if not message:
        message = "All specified columns were converted successfully."
    else:
        message = ' '.join(message)
    
    return df_converted, message


current_directory = os.getcwd()  # Get the current working directory
file_path = find_file_in_directory(current_directory)

df = createDataFrame(file_path)
# Show the description to the user, as the user will input column names when dropping them.
df, description = clean_column_names(df)
print(description)
column_string = "Column A, Column B, Invalid Column"
df, message = drop_columns_from_string(df, column_string)
print(message)
df, message = remove_duplicates(df)
print(message)
df,message = handle_missing_values(df)
print(message)
print(df.head())
