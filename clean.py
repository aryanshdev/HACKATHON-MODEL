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
    status=""
    if(file_extension==".csv"):
        df =pd.read_csv(path)
        message="Data Frame for .csv is created"
        status="succ"
    elif file_extension in [".xlsx",".xls"]:
        df = pd.read_excel(path, engine='openpyxl')
        message="Data frame for .xlsx or .xls is created"
        status="succ"
    else:
        message= "Unsupported File extension, please input csv, xlsx or xls file type."
        status="err"

    save_dataframe_to_csv(df)
    data = df.to_json(orient='records')
    result = {
        'data': data,
        'message': message,
        'status':status
    }
    return result

# takes df param and gives back df and a message
def clean_column_names(path):
    df =pd.read_csv(path)
    # Original column names
    original_columns = df.columns.tolist()
    # Clean column names
    df.columns = df.columns.str.strip()  # Remove leading and trailing whitespace
    df.columns = df.columns.str.lower()  # Convert to lowercase
    df.columns = df.columns.str.replace(r'[^a-z0-9]', '_', regex=True)  # Replace non-alphanumeric characters with underscores
    
     # Create description of changes
    cleaned_columns = df.columns.tolist()
    changes = []
    status=""
    
    for orig, cleaned in zip(original_columns, cleaned_columns):
        if orig != cleaned:
            changes.append(f"'{orig}' -> '{cleaned}'")
    
    if changes:
        message = "Column name changes:\n" + "\n".join(changes)
        status="succ"
    else:
        message = "No changes to column names."
        status="succ"

    save_dataframe_to_csv(df)
    data = df.to_json(orient='records')
    result = {
        'data': data,
        'message': message,
        'status':status
    }
    return result
    
# Takes in the dataframe and a String of the columns to be dropped; seperated by comma, returns df and message
def drop_columns_from_string(path, column_string):
    df =pd.read_csv(path)
    # Split the string by commas and trim whitespace
    columns_to_drop = [col.strip() for col in column_string.split(',')]
    status=""
    # Check for invalid column names
    invalid_columns = [col for col in columns_to_drop if col not in df.columns]
    
    if invalid_columns:
        # If there are invalid columns, return a message and do not modify the DataFrame
        message = f"Invalid column names: {', '.join(invalid_columns)}"
        status="err"
        data = df.to_json(orient='records')
        result = {
            'data': data,
            'message': message,
            'status':status
        }
        return result
        
    
    # Drop the specified columns from the DataFrame
    df_dropped = df.drop(columns=[col for col in columns_to_drop], errors='ignore')
    
    # Provide feedback on which columns were dropped
    dropped_columns = [col for col in columns_to_drop if col in df.columns]
    if dropped_columns:
        message = f"Dropped columns: {', '.join(dropped_columns)}"
        status="succ"
    else:
        message = "No columns were dropped."
        status="succ"
    
    save_dataframe_to_csv(df_dropped)
    data = df_dropped.to_json(orient='records')
    result = {
        'data': data,
        'message': message,
        'status':status
    }
    return result
  

# Removes duplicate rows and returns message as such.
def remove_duplicates(path):
    df =pd.read_csv(path)
    status=""
    # Check for duplicate rows
    if df.duplicated().any():
        # Remove duplicate rows
        df_cleaned = df.drop_duplicates()
        # Provide feedback on the number of duplicates removed
        num_duplicates = df.shape[0] - df_cleaned.shape[0]
        message = f"Removed {num_duplicates} duplicate row(s)."
        status="succ"
    else:
        # No duplicates found
        df_cleaned = df
        message = "No duplicate rows found."
        status="succ"

    save_dataframe_to_csv(df_cleaned)
    data = df_cleaned.to_json(orient='records')
    result = {
            'data': data,
            'message': message,
            'status':status
        }
    return result
        

# Takes input of df and returns output of df(DO NOT OVERWRITE ORIGINAL DF) and mssg
def check_missing_values(path):
    df =pd.read_csv(path)
    status=""
    # Calculate number of missing values and percentage for each column
    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    # Create a DataFrame with the results
    missing_summary = pd.DataFrame({
        'Missing Values': missing_counts,
        'Percentage': missing_percentages
    })
    data = missing_summary.to_json(orient='records')
    message="Consider dropping columns with high(>80%) missing values"
    status="succ"
    result = {
            'data': data,
            'message': message,
            'status':status
        }

    return result

# Takes df and column names of "NON-NUMERIC FIELDS", returns df and message
def handle_nonnumeric_missing_vals_fill(path, columns):
    df =pd.read_csv(path)
   
    columns_to_handle = [col.strip() for col in columns.split(',')]
    messages = []
    status=""
    
    for column in columns_to_handle:
        if column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna('Unknown')
                messages.append(f"Filled missing values in non-numeric column '{column}' with 'Unknown'.")
                status="succ"
            else:
                messages.append(f"Column '{column}' is numeric. Please use designated tab for such.")
                status="err"
        else:
            messages.append(f"Column '{column}' not found in DataFrame.")
            status="err"
    
    message = " ".join(messages)
    save_dataframe_to_csv(df)
    data = df.to_json(orient='records')
    result = {
            'data': data,
            'message': message,
            'status':status
        }
    return result

# Takes df and column names of "NON-NUMERIC FIELDS", return df and message
def handle_nonnumeric_missing_vals_drop(path, columns):
    df =pd.read_csv(path)
    message=""
    status=""
    columns_to_check = [col.strip() for col in columns.split(',')]
    
    # Ensure all specified columns are in the DataFrame
    if not all(col in df.columns for col in columns_to_check):
        missing_cols = [col for col in columns_to_check if col not in df.columns]
        message=f"Columns {', '.join(missing_cols)} not found in DataFrame."
        status="err"
        data = df.to_json(orient='records')
        result = {
                'data': data,
                'message': message,
                'status':status
            }
        return result
    
    # Count the number of rows with missing values in the specified columns before dropping
    initial_count = df.shape[0]
    
    # Drop rows where any of the specified columns have missing values
    df = df.dropna(subset=columns_to_check)
    
    # Count the number of rows dropped
    rows_dropped = initial_count - df.shape[0]
    message = f"Dropped {rows_dropped} rows where specified columns had missing values."
    status="succ"
    save_dataframe_to_csv(df)
    data = df.to_json(orient='records')
    result = {
            'data': data,
            'message': message,
            'status':status
        }
    return result
    
  


# Takes df and column names of "NUMERIC FILEDS", returns df and message
def handle_numeric_missing_vals(path, columns):
    df =pd.read_csv(path)

    columns_to_handle = [col.strip() for col in columns.split(',')]
    messages = []
    median_values = {}
    status=""
    for column in columns_to_handle:
        if column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
                median_values[column] = median_value
                messages.append(f"Filled missing values in numeric column '{column}' with mean value {median_value:.2f}.")
                status="succ"
            else:
                messages.append(f"Column '{column}' is non-numeric. Please use handle_nonnumeric_missing_vals for non-numeric columns.")
                status="err"
        else:
            messages.append(f"Column '{column}' not found in DataFrame.")
            status="err"
    
    message = " ".join(messages)
    save_dataframe_to_csv(df)
    data = df.to_json(orient='records')
    result = {
            'data': data,
            'message': message,
            'status':status
        }
    return result


# Takes df and column names and returns df and message
def convert_to_numeric(path, columns):
    df =pd.read_csv(path)
    
    # Split the comma-separated string into a list of columns
    column_list = [col.strip() for col in columns.split(',')]
    
    message = []
    status=""
    # Copy the DataFrame to avoid modifying the original data
    df_converted = df.copy()
    
    for column in column_list:
        if column in df_converted.columns:
            try:
                # Attempt to convert the column to numeric
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
                message.append(f"Column '{column}' converted to numeric.")
                status="succ"
            except Exception as e:
                message.append(f"Error converting column '{column}': {e}")
                status="err"
        else:
            message.append(f"Column '{column}' not found in DataFrame.")
            status="err"
    
    # Check if there were any issues with conversion
    if not message:
        message = "All specified columns were converted successfully."
        status="succ"
    else:
        message = ' '.join(message)
    
    save_dataframe_to_csv(df_converted)
    data = df_converted.to_json(orient='records')
    result = {
            'data': data,
            'message': message,
            'status':status
        }

    return result

# Creates a file and returns back a custom message 
def save_dataframe_to_csv(df):
    try:
        # Save the DataFrame to 'clean.csv'
        df.to_csv('clean.csv', index=False)
        return "File 'clean.csv' has been successfully created."
    except Exception as e:
        return f"An error occurred while creating the file 'clean.csv': {e}"

# Normalize ONLY DATE values and returns df and custom message
def normalize_date_column(path, date_column):
    df =pd.read_csv(path)
  
    if date_column not in df.columns:
        data = df.to_json(orient='records')
        message=f"Column '{date_column}' not found in DataFrame."
        status="err"
        result = {
            'data': data,
            'message': message,
            'status':status
        }
        return result
    
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

        save_dataframe_to_csv(df)
        data = df.to_json(orient='records')
        message=f"Date column '{date_column}' normalized successfully."
        status="succ"
        result = {
            'data': data,
            'message': message,
            'status':status
        }
        
        return result
    
    except Exception as e:
        data = df.to_json(orient='records')
        message=f"An error occurred while normalizing the date column '{date_column}': {e}"
        status="err"
        result = {
            'data': data,
            'message': message,
            'status':status
        }
        return result

# Input a df and target_col name and returns df and message
def one_hot_encoding(path, target_column):
    df =pd.read_csv(path)
    
    # Check if target_column is in the DataFrame
    if target_column not in df.columns:
        data = df.to_json(orient='records')
        message=f"Target column '{target_column}' not found in DataFrame."
        status="err"
        result = {
            'data': data,
            'message': message,
            'status':status
        }
        return result

    
    # Extract the target column
    df_target = df[target_column]
    
    # Drop the target column from the DataFrame
    df = df.drop(columns=[target_column])
    
    # Select categorical columns for one-hot encoding
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_columns:
        df[target_column] = df_target.values 
        data = df.to_json(orient='records')
        message="No categorical columns found for one-hot encoding."
        status="err"
        result = {
            'data': data,
            'message': message,
            'status':status
        }
        return result

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
    save_dataframe_to_csv(df_encoded)
    data = df_encoded.to_json(orient='records')
    message=f"One-Hot Encoding applied to columns: {', '.join(categorical_columns)}. Target column '{target_column}' is at the end."
    status="succ"
    result = {
            'data': data,
            'message': message,
            'status':status
        }
    
    return result


# Takes df from user and returns a df_type(NOT SAME AS INPUT df; DO NOT OVERWRITE INPUT DF)
def get_column_datatypes(path):
    df =pd.read_csv(path)
    column_dtypes = df.dtypes
    data = column_dtypes.to_json(orient='records')
    message="Following is the listed data types of each column. Please ensure to have a numeric only dataset(excluding target value)"
    status="succ"
    result = {
            'data': data,
            'message': message,
            'status':status
        }
    return result



# takes in df, target_column String and returns df and message
def drop_rows_without_target(path, target_column):
    df =pd.read_csv(path)
 
    if target_column not in df.columns:
        data = df.to_json(orient='records')
        message=f"Target column '{target_column}' not found in DataFrame."
        status="err"
        result = {
            'data': data,
            'message': message,
            'status':status
        }
        return result
    
    # Count the number of rows with missing target values before dropping
    initial_count = df.shape[0]
    
    # Drop rows where the target column has missing values
    df_cleaned = df.dropna(subset=[target_column])
    
    # Count the number of rows dropped
    rows_dropped = f"Rows dropped count: {initial_count - df_cleaned.shape[0]}"

    save_dataframe_to_csv(df_cleaned)
    data = df_cleaned.to_json(orient='records')
    message=rows_dropped
    status="succ"
    result = {
            'data': data,
            'message': message,
            'status':status
        }
    
    return result
    
    return df_cleaned, rows_dropped

# --------------------------------TESTING CODE-------------------------

current_directory = os.getcwd()  # Get the current working directory



# file_path="uncleaned bike sales data.xlsx"
# df,message = createDataFrame(file_path)
# print(message)

# print("Cleaning column names")
# df,message = clean_column_names(df)
# print(message)
# print(df)

# print("Dealing with duplicate entries")
# df,message=remove_duplicates(df)
# print(df)
# print(message)

# missing_summary_df,message=check_missing_values(df)
# print(missing_summary_df)
# print(message)

# nonNumeric=input("Enter the Non-Numeric columns(FILL): ")
# df,message=fill_handle_nonnumeric_missing_vals(df,nonNumeric)
# print(df)
# print(message)

# missing_summary_df,message=check_missing_values(df)
# print(missing_summary_df)
# print(message)


# nonNumeric=input("Enter the Non-Numeric columns(DROP): ")
# df,message=drop_rows_with_nonnumeric_missing_vals(df,nonNumeric)
# print(df)
# print(message)

# missing_summary_df,message=check_missing_values(df)
# print(missing_summary_df)
# print(message)


# nonNumericColumns=input("Enter the Numeric columns: ")
# df,message=handle_numeric_missing_vals(df,nonNumericColumns)
# print(df)
# print(message)

# missing_summary_df,message=check_missing_values(df)
# print(missing_summary_df)
# print(message)


# columnsToBeDropped=input("Enter the columns seperated by comma, that needs to be dropped.")
# df,message=drop_columns_from_string(df,columnsToBeDropped)
# print(df)
# print(message)

# date_column=input("Enter the date column: ")
# df,message=normalize_date_column(df,date_column)
# print(df)
# print(message)

# df_type=get_column_datatypes(df);
# print(df_type)


# target=input("OHE: enter target variable ")
# df,message=one_hot_encoding(df,target)
# print(df)
# print(message)


# df_type=get_column_datatypes(df);
# print(df_type)

# df,message=drop_rows_without_target(df,input("Enter target column name "))
# print(df)
# print(message)

# columnnnames=input("Enter the columns that need to be converted to numeric ")
# df,message=convert_to_numeric(df,columnnnames)
# print(df)
# print(message)

# df_type=get_column_datatypes(df);
# print(df_type)

# print(df)
# ṇ