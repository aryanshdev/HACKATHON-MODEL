import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


# Takes in path, returns df and message
def createDataFrame(path):

    file_extension = os.path.splitext(path)[1]
    df=None
    message=""
    if(file_extension==".csv"):
        df =pd.read_csv(path)
        message="Data Frame for .csv is created"
    elif file_extension in [".xlsx",".xls"]:
        df = pd.read_excel(path)
        message="Data frame for .xlsx or .xls is created"
    else:
        message= "Unsupported File extension, please input csv, xlsx or xls file type."

    return df,message

# Can only return the 1st valid file it encounters.
def find_file_in_directory(directory, extensions= ['.csv', '.xlsx', '.xls']):
    # List all files in the directory
    for file_name in os.listdir(directory):
        # Check if the file has the desired extension
        if any(file_name.endswith(ext) for ext in extensions):
            return os.path.join(directory, file_name)
    return None

# takes df param and gives back df and a message
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
        message = "Column name changes:\n" + "\n".join(changes)
    else:
        message = "No changes to column names."
    
    return df, message


# Takes in the dataframe and a String of the columns to be dropped; seperated by comma, returns df and message
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


# Takes input of df and returns output of df
def check_missing_values(df):
    """
    Check missing values in each column of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame containing the number of missing values and percentage of missing values for each column.
    """
    # Calculate number of missing values and percentage for each column
    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    # Create a DataFrame with the results
    missing_summary = pd.DataFrame({
        'Missing Values': missing_counts,
        'Percentage': missing_percentages
    })
    message="Consider dropping columns with high(>80%) missing values"
    return missing_summary,message

# Takes df and column names of "NON-NUMERIC FIELDS", returns df and message
def handle_nonnumeric_missing_vals(df, columns):
    """
    Fill missing values in a specified non-numeric column with 'Unknown'.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The name of the non-numeric column to process.
    
    Returns:
    pd.DataFrame: DataFrame with missing values in the specified column filled with 'Unknown'.
    str: Message indicating the result of the operation.
    """
    columns_to_handle = [col.strip() for col in columns.split(',')]
    messages = []
    
    for column in columns_to_handle:
        if column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna('Unknown')
                messages.append(f"Filled missing values in non-numeric column '{column}' with 'Unknown'.")
            else:
                messages.append(f"Column '{column}' is numeric. Please use handle_numeric_missing_vals for numeric columns.")
        else:
            messages.append(f"Column '{column}' not found in DataFrame.")
    
    message = " ".join(messages)
    return df, message

# Takes df and column names of "NUMERIC FILEDS", returns df and message
def handle_numeric_missing_vals(df, columns):
    """
    Fill missing values in specified numeric columns with the mean of each column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (str): Comma-separated string of column names to process.
    
    Returns:
    pd.DataFrame: DataFrame with missing values in the specified columns filled with the mean value.
    str: Message indicating the result of the operation and the mean values used.
    dict: Dictionary with column names and their mean values.
    """
    columns_to_handle = [col.strip() for col in columns.split(',')]
    messages = []
    mean_values = {}
    
    for column in columns_to_handle:
        if column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                mean_value = df[column].mean()
                df[column] = df[column].fillna(mean_value)
                mean_values[column] = mean_value
                messages.append(f"Filled missing values in numeric column '{column}' with mean value {mean_value:.2f}.")
            else:
                messages.append(f"Column '{column}' is non-numeric. Please use handle_nonnumeric_missing_vals for non-numeric columns.")
        else:
            messages.append(f"Column '{column}' not found in DataFrame.")
    
    message = " ".join(messages)
    return df, message


# Takes df and column names and returns df and message
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

# Creates a file and returns back a custom message 
def save_dataframe_to_csv(df):
    """
    Save the DataFrame to a CSV file named 'clean.csv' and return a message.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to save to a CSV file.
    
    Returns:
    str: Message indicating the result of the file creation process.
    """
    try:
        # Save the DataFrame to 'clean.csv'
        df.to_csv('clean.csv', index=False)
        return "File 'clean.csv' has been successfully created."
    except Exception as e:
        return f"An error occurred while creating the file 'clean.csv': {e}"

# Normalize ONLY DATE values and returns df and custom message
def normalize_date_column(df, date_column):
    """
    Normalize the date column in the DataFrame such that:
    - The minimum date is mapped to 0.
    - The maximum date is mapped to 1.
    - All other dates are scaled between 0 and 1 based on their relative position.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the date column to normalize.
    date_column (str): The name of the column containing date values.
    
    Returns:
    pd.DataFrame: DataFrame with the normalized date column.
    str: Message indicating the result of the normalization process.
    """
    if date_column not in df.columns:
        return df, f"Column '{date_column}' not found in DataFrame."
    
    try:
        # Convert the date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Find the minimum and maximum date
        min_date = df[date_column].min()
        max_date = df[date_column].max()
    
        # Calculate the range in days
        range_days = (max_date - min_date).days
        
        # Convert dates to the number of days since the minimum date
        df[date_column] = (df[date_column] - min_date).dt.days
        
        # Normalize the date column between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[[date_column]] = scaler.fit_transform(df[[date_column]])
        
        return df, f"Date column '{date_column}' normalized successfully."
    
    except Exception as e:
        return df, f"An error occurred while normalizing the date column '{date_column}': {e}"

# Input a df and target_col name and returns df and message
def one_hot_encoding(df, target_column):
    """
    Apply One-Hot Encoding to categorical columns and move the target column to the end of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column to keep at the end of the DataFrame.
    
    Returns:
    pd.DataFrame: Updated DataFrame with One-Hot Encoded columns and target column at the end.
    str: Message indicating the result of the operation.
    """
    # Check if target_column is in the DataFrame
    if target_column not in df.columns:
        return df, f"Target column '{target_column}' not found in DataFrame."
    
    # Extract the target column
    df_target = df[target_column]
    
    # Drop the target column from the DataFrame
    df = df.drop(columns=[target_column])
    
    # Select categorical columns for one-hot encoding
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_columns:
        df[target_column] = df_target.values 
        return df, f"No categorical columns found for one-hot encoding."

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    
    # Apply one-hot encoding to the categorical columns
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])
    
    # Create a DataFrame with the one-hot encoded columns
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Concatenate the one-hot encoded dataframe with the original dataframe (excluding categorical columns)
    df_encoded = pd.concat([df.drop(categorical_columns, axis=1), one_hot_df], axis=1)
    
    # Add the target column back to the end
    df_encoded[target_column] = df_target.values  # Ensure proper alignment
    
    return df_encoded, f"One-Hot Encoding applied to columns: {', '.join(categorical_columns)}. Target column '{target_column}' is at the end."


# Takes df from user and returns a df_type(NOT SAME AS INPUT df; DO NOT OVERWRITE INPUT DF)
def get_column_datatypes(df):
    """
    Get the data type of each column in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.Series: A Series with column names as the index and their data types as the values.
    """
    # Get data types of each column
    column_dtypes = df.dtypes
    
    return column_dtypes


# takes in df, target_column String and returns df and message
def drop_rows_without_target(df, target_column):
    """
    Drop rows where the target column has missing values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    
    Returns:
    pd.DataFrame: DataFrame with rows dropped where the target column has missing values.
    int: The count of rows that were dropped.
    """
    if target_column not in df.columns:
        return df, f"Target column '{target_column}' not found in DataFrame."
    
    # Count the number of rows with missing target values before dropping
    initial_count = df.shape[0]
    
    # Drop rows where the target column has missing values
    df_cleaned = df.dropna(subset=[target_column])
    
    # Count the number of rows dropped
    rows_dropped = f"Rows dropped count: {initial_count - df_cleaned.shape[0]}"
    
    return df_cleaned, rows_dropped

# --------------------------------TESTING CODE-------------------------

current_directory = os.getcwd()  # Get the current working directory
file_path = find_file_in_directory(current_directory)

file_path="fetal_health_modified_2.csv"
df,message = createDataFrame(file_path)
print(message)

df,message = clean_column_names(df)
print(message)
print(df)
# Column name changes:
# 'ID' -> 'id'
# 'Name' -> 'name'
# 'Age' -> 'age'
# 'Salary' -> 'salary'
# 'Department' -> 'department'
# 'JoiningDate' -> 'joiningdate'
# 'Location' -> 'location'
# 'Gender' -> 'gender'
# 'Experience' -> 'experience'
# 'PerformanceRating' -> 'performancerating'


missing_summary_df,message=check_missing_values(df)
print(missing_summary_df)
print(message)

# numericColumns=input("Enter the Numeric columns: ")
# df,message=handle_nonnumeric_missing_vals(df,numericColumns)
# print(df)
# print(message)


# nonNumericColumns=input("Enter the Non-Numeric columns: ")
# df,message=handle_numeric_missing_vals(df,nonNumericColumns)
# print(df)
# print(message)

columnsToBeDropped=input("Enter the columns seperated by comma, that needs to be dropped.")
df,message=drop_columns_from_string(df,columnsToBeDropped)
print(df)
print(message)

# df,message=normalize_date_column(df,"joiningdate")
# print(df)
# print(message)

df_type=get_column_datatypes(df);
print(df_type)

df,message=one_hot_encoding(df,"fetal_health")
print(df)
print(message)


df_type=get_column_datatypes(df);
print(df_type)

df,message=drop_rows_without_target(df,"fetal_health")
print(df)
print(message)

df,message=convert_to_numeric(df,"fetal_health")
print(df)
print(message)

df_type=get_column_datatypes(df);
print(df_type)

df,message=remove_duplicates(df)
print(df)
print(message)
