import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,mean_squared_error
import joblib
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import pickle
import os
from sklearn.preprocessing import LabelEncoder



def save_target_variable(path,target_column,code=""):
    # Load the DataFrame
    df = pd.read_csv(path)
    data = None
    message = ""
    status = ""
    file_name=f"intermediate/{code}_target_var_label_enc_status.pickle"

    # Ensure the intermediate folder exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

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


def update_encoder_status(str_val,code=""):
    data = None
    status = ""
    file_name=f"intermediate/{code}_target_var_label_enc_status.pickle"
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
        with open(file_name, 'wb') as file:
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
   
    
    

def read_target_var_label_enc_status(code=""):
    file_path=f"intermediate/{code}_target_var_label_enc_status.pickle"
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
    file_path=f"intermediate/{code}_target_var_label_enc_status.pickle"
    target_column=read_target_var_label_enc_status(code)[0][1]
    print(target_column)
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

        os.makedirs(os.path.dirname(f"intermediate/{code}_XY_train.csv"), exist_ok=True)
        os.makedirs(os.path.dirname(f"intermediate/{code}_XY_test.csv"), exist_ok=True)

        # Concatenate features and target
        train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        # Save to CSV files
        train_df.to_csv(f'intermediate/{code}_XY_train.csv', index=False)
        test_df.to_csv(f'intermediate/{code}_XY_test.csv', index=False)
        

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


def random_forest(target,enco_status,df_train,df_test,n_estimators,max_depth,min_samples_split,code=""):
    n_estimators=int(n_estimators)
    if(max_depth!="None"):
        max_depth=int(max_depth)
    else:
        max_depth=None
    min_samples_split=int(min_samples_split)

    status = "success"
    message = ""
    data = None
    try:
        if enco_status==0:

            X_train = df_train.drop(columns=[target])  # Features
            y_train = df_train[target] 

            X_test = df_test.drop(columns=[target])
            y_test=df_test[target]

            rf_clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split)
            rf_clf.fit(X_train, y_train)

            y_train_pred = rf_clf.predict(X_train)
            y_test_pred = rf_clf.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            model_filename = f"output/{code}_random_forest.pkl"
            os.makedirs('output', exist_ok=True)
            joblib.dump(rf_clf, model_filename)
            message = f"Trained model saved as {model_filename}"
            

        else:
            le = LabelEncoder()
            df_train[target] = le.fit_transform(df_train[target])
            df_test[target] = le.transform(df_test[target])

            X_train = df_train.drop(columns=[target])
            y_train = df_train[target]
            X_test = df_test.drop(columns=[target])
            y_test = df_test[target]
            
            rl_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
            rl_clf.fit(X_train, y_train)

            # Calculate train and test accuracy
            train_accuracy = accuracy_score(y_train, rl_clf.predict(X_train))
            test_accuracy = accuracy_score(y_test, rl_clf.predict(X_test))

            # Save the model
            model_filename = f"output/{code}_enco_random_forest.pkl"
            os.makedirs('output', exist_ok=True)
            joblib.dump(rl_clf, model_filename)
            message = f"Trained model saved as {model_filename}"
        
        return {
            'data': {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            },
            'status': status,
            'message': message
        }
    
    except Exception as e:
        # Handle exceptions and return error message
        return {
            'data': None,
            'status': "error",
            'message': f"An error occurred: {e}"
        }
        

def xgboost(target,enco_status,df_train,df_test,n_estimators,max_depth,learning_rate,code=""):
    n_estimators=int(n_estimators)
    max_depth=int(max_depth)
    learning_rate=float(learning_rate)

    status = "success"
    message = ""
    data = None
    try:
        if enco_status==0:
            X_train = df_train.drop(columns=[target])
            y_train = df_train[target]
            X_test = df_test.drop(columns=[target])
            y_test = df_test[target]

            xg_clf = xgboost.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            model_filename = f"output/{code}_xgboost_model.pkl"
            os.makedirs('output', exist_ok=True)
            joblib.dump(xg_clf, model_filename)
            message=f"Trained model saved as {model_filename}"

        else:
            le = LabelEncoder()
            df_train[target] = le.fit_transform(df_train[target])
            df_test[target] = le.transform(df_test[target])

            X_train = df_train.drop(columns=[target])
            y_train = df_train[target]
            X_test = df_test.drop(columns=[target])
            y_test = df_test[target]


            xg_clf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            xg_clf.fit(X_train, y_train)

            y_train_pred = xg_clf.predict(X_train)
            y_test_pred = xg_clf.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            model_filename = f"output/{code}_enco_xgboost_model.pkl"
            os.makedirs('output', exist_ok=True)
            joblib.dump(xg_clf, model_filename)
            message=f"Trained model saved as {model_filename}"

        return {
            'data': {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            },
            'status': status,
            'message': message
        }
    
    except Exception as e:
        # Handle exceptions and return error message
        return {
            'data': None,
            'status': "error",
            'message': f"An error occurred: {e}"
        }


def bagging(target,enco_status,df_train,df_test,n_estimators,max_samples,max_features,code=""):
    n_estimators=int(n_estimators)
    max_samples=int(max_samples)
    max_features=float(max_features)

    status = "success"
    message = ""
    data = None
    try:
        if enco_status==0:
            X_train = df_train.drop(columns=[target])
            y_train = df_train[target]
            X_test = df_test.drop(columns=[target])
            y_test = df_test[target]

            base_estimator = DecisionTreeClassifier()
            bag_clf = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=42)
            bag_clf.fit(X_train, y_train)

            y_train_pred = bag_clf.predict(X_train)
            y_test_pred = bag_clf.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)


            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            model_filename = f"output/{code}_bagging_model.pkl"
            os.makedirs('output', exist_ok=True)
            joblib.dump(bag_clf, model_filename)
            message = f"Trained model saved as {model_filename}"

        else:
            le = LabelEncoder()
            df_train[target] = le.fit_transform(df_train[target])
            df_test[target] = le.transform(df_test[target])

            X_train = df_train.drop(columns=[target])
            y_train = df_train[target]
            X_test = df_test.drop(columns=[target])
            y_test = df_test[target]

            base_estimator = DecisionTreeClassifier()
            bag_clf = BaggingClassifier(estimator=base_estimator, n_estimators=n_estimators, max_samples=max_samples,max_features=max_features,random_state=42)
            bag_clf.fit(X_train, y_train)

            y_train_pred = bag_clf.predict(X_train)
            y_test_pred = bag_clf.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)


            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            model_filename = f"output/{code}_enco_bagging_model.pkl"
            os.makedirs('output', exist_ok=True)
            joblib.dump(bag_clf, model_filename)
            message = f"Trained model saved as {model_filename}"
        return {
            'data': {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            },
            'status': status,
            'message': message
        }
    
    except Exception as e:
        # Handle exceptions and return error message
        return {
            'data': None,
            'status': "error",
            'message': f"An error occurred: {e}"
        }


# Define dropdown for [kernel poly, rbf, sigmoid] and text box for gamma and c
def svm(target,enco_status,df_train,df_test,kernel,c,gamma,code=""):
    c=float(c)
    gamma=float(gamma)
    kernel=kernel

    status = "success"
    message = ""
    data = None
    try:
        if enco_status==0:

            X_train = df_train.drop(columns=[target])  # Features
            y_train = df_train[target] 

            X_test = df_test.drop(columns=[target])
            y_test=df_test[target]

            svm_clf = SVC(kernel=kernel,C=c,gamma=gamma)
            svm_clf.fit(X_train, y_train)

            y_train_pred = svm_clf.predict(X_train)
            y_test_pred = svm_clf.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            model_filename = f"output/{code}_svm.pkl"
            os.makedirs('output', exist_ok=True)
            joblib.dump(svm_clf, model_filename)
            message = f"Trained model saved as {model_filename}"
            

        else:
            le = LabelEncoder()
            df_train[target] = le.fit_transform(df_train[target])
            df_test[target] = le.transform(df_test[target])

            X_train = df_train.drop(columns=[target])
            y_train = df_train[target]
            X_test = df_test.drop(columns=[target])
            y_test = df_test[target]
            
            svm_clf = SVC(kernel=kernel,C=c,gamma=gamma)
            svm_clf.fit(X_train, y_train)

            # Calculate train and test accuracy
            train_accuracy = accuracy_score(y_train, svm_clf.predict(X_train))
            test_accuracy = accuracy_score(y_test, svm_clf.predict(X_test))

            # Save the model
            model_filename = f"output/{code}_enco_svm.pkl"
            os.makedirs('output', exist_ok=True)
            joblib.dump(svm_clf, model_filename)
            message = f"Trained model saved as {model_filename}"
        
        return {
            'data': {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            },
            'status': status,
            'message': message
        }
    
    except Exception as e:
        # Handle exceptions and return error message
        return {
            'data': None,
            'status': "error",
            'message': f"An error occurred: {e}"
        }
        
# Define input for all[1,2,3,4,5...any no]
def decision_tree(target,enco_status,df_train,df_test,max_depth,min_samples_split,criterion,code=""):
    max_depth=int(max_depth)
    min_samples_split=int(min_samples_split)

    status = "success"
    message = ""
    data = None
    try:
        if enco_status==0:

            X_train = df_train.drop(columns=[target])  # Features
            y_train = df_train[target] 

            X_test = df_test.drop(columns=[target])
            y_test=df_test[target]

            svm_clf = DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,criterion=criterion)
            svm_clf.fit(X_train, y_train)

            y_train_pred = svm_clf.predict(X_train)
            y_test_pred = svm_clf.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            model_filename = f"output/{code}_decision_tree.pkl"
            os.makedirs('output', exist_ok=True)
            joblib.dump(svm_clf, model_filename)
            message = f"Trained model saved as {model_filename}"
            

        else:
            le = LabelEncoder()
            df_train[target] = le.fit_transform(df_train[target])
            df_test[target] = le.transform(df_test[target])

            X_train = df_train.drop(columns=[target])
            y_train = df_train[target]
            X_test = df_test.drop(columns=[target])
            y_test = df_test[target]
            
            svm_clf = DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,criterion=criterion)
            svm_clf.fit(X_train, y_train)

            # Calculate train and test accuracy
            train_accuracy = accuracy_score(y_train, svm_clf.predict(X_train))
            test_accuracy = accuracy_score(y_test, svm_clf.predict(X_test))

            # Save the model
            model_filename = f"output/{code}_enco_decision_tree.pkl"
            os.makedirs('output', exist_ok=True)
            joblib.dump(svm_clf, model_filename)
            message = f"Trained model saved as {model_filename}"
        
        return {
            'data': {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            },
            'status': status,
            'message': message
        }
    
    except Exception as e:
        # Handle exceptions and return error message
        return {
            'data': None,
            'status': "error",
            'message': f"An error occurred: {e}"
        }
        


def model(model_name,param1,param2,param3,train_file='XY_train.csv',test_file='XY_test.csv',code=''):
    # Load data
    file_name=f'intermediate/{code}_target_var_label_enc_status.pickle'
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    enco_status, target = read_target_var_label_enc_status(code)[0]
    code=""
    # enco_status is a string




    if model_name == 'random_forest':
        response = random_forest(target,enco_status,df_train,df_test,param1,param2,param3,code)
    elif model_name == 'xgboost':
       response = xgboost(target,enco_status,df_train,df_test,param1,param2,param3,code)
    elif model_name == 'bagging':
        response  = bagging(target,enco_status,df_train,df_test,param1,param2,param3,code)
    elif model_name == 'svm':
        response = svm(target,enco_status,df_train,df_test,param1,param2,param3,code)
    elif model_name == 'decision_tree':
        response = decision_tree(target,enco_status,df_train, df_test,param1,param2,param3,code)
    else:
        raise ValueError(f"Unknown model code: {model_name}")

    return response


# ---------------------------------------------------TESTING SPLITS------------------------------------------------------#


# target_var = input("Enter target varaible name: ")
# path="fetal_health_encoded.csv"
# save_target_variable(path,target_var)


# str=input("Enter 'yes' if target values are categorigal data and 'no' for non categorical data: ")
# update_encoder_status(str)

# trainpercen=input("Enter the % of training data: ")
# response=split_and_save_data(path,trainpercen)
# # print(response)


# ---------------------------------------------------TESTING MODELS------------------------------------------------------#

# model_name='xgboost'
# param1='100'
# param2='3'
# param3='0.5'
# response=model(model_name,param1,param2,param3)
# print(response)


# model_name='bagging'
# param1='100'
# param2='10'
# param3='0.5'
# response=model(model_name,param1,param2,param3)
# print(response)

# model_name='random_forest'
# param1='100'
# param2='None'
# param3='2'
# response=model(model_name,param1,param2,param3)
# print(response)

# # param1=kernel only rbf
# # param2= C [10^-3 to 10^3]
# #param3= gamma[same]
# model_name='svm'
# param1="rbf" #LOCKED VALUE
# param2="10"
# param3="10"
# response=model(model_name,param1,param2,param3)
# print(response)

# # param1=maxdepth only numbers
# # param2= min_samples_split only nos
# #param3= criterion ['gini', 'entropy', 'log_loss']
# model_name='decision_tree'
# param1="10"
# param2="10"
# param3="entropy"
# response=model(model_name,param1,param2,param3)
# print(response)