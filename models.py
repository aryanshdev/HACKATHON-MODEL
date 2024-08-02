import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

def save_target_variable(path,target_column,file_name='target_var_label_enc_status.pickle'):
    # Load the DataFrame
    df = pd.read_csv(path)
    data = None
    message = ""
    status = ""

    if target_column not in df.columns:
        message = f"Target column '{target_column}' not found in DataFrame."
        status = "err"
        result = {
            'data': data,
            'message': message,
            'status': status
        }
        return result

    try:
        # Save the target column name to a pickle file
        with open(file_name, 'wb') as file:
            pickle.dump(target_column, file)
        
        message = f"Target column name '{target_column}' has been successfully saved to '{file_name}'."
        status = "succ"
        result = {
            'data': data,
            'message': message,
            'status': status
        }
        return result
    except Exception as e:
        message = f"An error occurred while saving the target column name: {e}"
        status = "err"
        result = {
            'data': data,
            'message': message,
            'status': status
        }
        return result
    


# def load_target_variable(file_name='target_var_label_enc_status.pickle'):
#     try:
#         # Load the target variable from the pickle file
#         with open(file_name, 'rb') as file:
#             target_values = pickle.load(file)
#         return target_values, f"Target variable has been successfully loaded from '{file_name}'."
#     except Exception as e:
#         return None, f"An error occurred while loading the target variable: {e}"
    

def update_encoder_status(str_val,file_name='target_var_label_enc_status.pickle'):
    data = None
    status = ""
    
    if str_val.lower() not in ["yes", "no"]:
        status = "err"
        message = "Invalid string value. Please use 'yes' or 'no'."
        result = {
            'data': data,
            'message': message,
            'status': status
        }
        return result
    
    try:
        # Load the target column name from the pickle file
        with open(file_name, 'rb') as file:
            target_column_name = pickle.load(file)
        
        # Determine the status based on str_val
        encoder_status = 1 if str_val.lower() == "yes" else 0

        # Save the encoder status and target column name to the new pickle file
        with open('target_var_label_enc_status.pickle', 'wb') as file:
            pickle.dump((encoder_status, target_column_name), file)
        
        status = "succ"
        message = f"Encoder status '{encoder_status}' and target column name have been successfully saved to 'target_var_label_enc_status.pickle'."
        result = {
            'data': data,
            'message': message,
            'status': status
        }
        return result
    
    except Exception as e:
        status = "err"
        message = f"An error occurred: {e}"
        result = {
            'data': data,
            'message': message,
            'status': status
        }
        return result
   
    
    

def read_target_var_label_enc_status(file_path='target_var_label_enc_status.pickle'):

    try:
        # Load the status and target variable from the pickle file
        with open(file_path, 'rb') as file:
            status, target_values = pickle.load(file)
        
        return (status, target_values), f"Data has been successfully loaded from '{file_path}'."
    except Exception as e:
        return None, f"An error occurred while loading the data: {e}"
    


def split_and_save_data(path, train_size_percentage, file_path='target_var_label_enc_status.pickle',code=''):
    data= None
    status="err"
    target_column=read_target_var_label_enc_status(file_path)[0][1]
    train_size_percentage=int(train_size_percentage)
    if not 0 < train_size_percentage < 100:
        message="Error: Training size percentage must be between 0 and 100."
        result = {
        'data': data,
        'message': message,
        'status':status
        }
        return result
    df = pd.read_csv(path)
    try:
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=train_size_percentage / 100, 
            random_state=42
        )

        # Concatenate features and target
        train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        # Save to CSV files
        train_df.to_csv(f'{code}XY_train.csv', index=False)
        test_df.to_csv(f'{code}XY_test.csv', index=False)
        

        status="succ"
        message="Data has been successfully split and saved to CSV files."
        result = {
        'data': data,
        'message': message,
        'status':status
        }
        return result

    except Exception as e:
        status="err"
        message=f"An error occurred: {e}"
        result = {
        'data': data,
        'message': message,
        'status':status
        }
        return result


target_var = input("Enter target varaible name: ")
path="fetal_health.csv"
save_target_variable(path,target_var)


str=input("Enter 'yes' if target values are categorigal data and 'no' for non categorical data: ")
update_encoder_status(str)

trainpercen=input("Enter the % of training data: ")
response=split_and_save_data(path,trainpercen)
# print(response)