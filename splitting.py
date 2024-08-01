import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_user_input():
    # Get training and testing percentages from the user
    train_size = float(input("Enter the training data percentage (e.g., 80 for 80%): ")) / 100
    test_size = 1 - train_size
    return train_size, test_size

def find_and_split_file(directory):
    # Define the desired file name
    file_name = "clean.csv"
    file_path = os.path.join(directory, file_name)
    
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        
        # Get the test percentage from the user
        train_size, test_size = get_user_input()
        
        print(f"Training data percentage: {train_size * 100:.2f}%")
        print(f"Testing data percentage: {test_size * 100:.2f}%")

        # Split the data into training and testing sets
        X = data.drop('target', axis=1)  # Features
        y = data['target']  # Target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Create dataframes for training and testing sets
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # Save the training and testing sets to separate CSV files
        train_data.to_csv("X_train_Y_train.csv", index=False)
        test_data.to_csv("X_test_Y_test.csv", index=False)
        
        print("Training and testing sets have been saved to X_train_Y_train.csv and X_test_Y_test.csv")
    else:
        print("The file clean.csv does not exist in the specified directory")

# Call the function with the parent directory
find_and_split_file("..") # Parent directory path
