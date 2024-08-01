import pandas as pd
import warnings
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

warnings.filterwarnings('ignore')


def load_data(file_name):
    try:
        data = pd.read_csv(file_name)
        return data
    except Exception as e:
        print(f"Error occurred while loading the data from {file_name}: ", str(e))


def train_xgboost(train_data):
    try:
        # Split the data into features and target variable
        X_train = train_data.drop(columns=['fetal_health'])  # Features
        y_train = train_data['fetal_health']  # Target variable

        # Define parameters for the XGBoost
        n_estimators = 100
        learning_rate = 0.1
        max_depth = 3
        objective = 'binary:logistic'

        # Initialize and train the XGBoost classifier
        xgb_clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, objective=objective, random_state=42)
        xgb_clf.fit(X_train, y_train)

        # Make predictions on the training data
        y_train_pred = xgb_clf.predict(X_train)

        return xgb_clf, n_estimators, learning_rate, max_depth, objective, X_train, y_train, y_train_pred
    except Exception as e:
        print("Error occurred while training the XGBoost classifier: ", str(e))


def evaluate_classifier(clf, n_estimators, learning_rate, max_depth, objective, X_train, y_train, y_train_pred, X_test, y_test):
    try:
        # Make predictions on the testing data
        y_test_pred = clf.predict(X_test)

        # Evaluate the model's performance
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        train_report = classification_report(y_train, y_train_pred)
        test_report = classification_report(y_test, y_test_pred)

        train_matrix = confusion_matrix(y_train, y_train_pred)
        test_matrix = confusion_matrix(y_test, y_test_pred)

        # Output the parameters and performance
        print(f"N_ESTIMATORS: {n_estimators}")
        print(f"LEARNING_RATE: {learning_rate}")
        print(f"MAX_DEPTH: {max_depth}")
        print(f"OBJECTIVE: {objective}")
        print(f"TRAINING ACCURACY: {train_accuracy * 100:.2f}%")
        print(f"TESTING ACCURACY: {test_accuracy * 100:.2f}%")
        print("\nTraining Classification Report:\n", train_report)
        print("\nTesting Classification Report:\n", test_report)
        print("\nTraining Confusion Matrix:\n", train_matrix)
        print("\nTesting Confusion Matrix:\n", test_matrix)

        # Save the trained model to a file
        model_filename = 'xgboost_model.pkl'
        joblib.dump(clf, model_filename)
        print(f"Trained model saved as {model_filename}")
    except Exception as e:
        print("Error occurred while evaluating the XGBoost classifier: ", str(e))


if __name__ == "__main__":
    # Define the paths to the training and testing files
    X_train_file = 'X_train.csv'
    X_test_file = 'X_test.csv'

    Y_train_file = 'y_train.csv'
    Y_test_file = 'y_test.csv'
    
    # Load data
    X_train_data = load_data(X_train_file)
    X_test_data = load_data(X_test_file)

    y_train_data = load_data(Y_train_file)
    y_test_data = load_data(Y_test_file)

    train_data = pd.concat([X_train_data, y_train_data], axis=1)
    test_data = pd.concat([X_test_data, y_test_data], axis=1)

    # Train the XGBoost classifier
    xgb_clf, n_estimators, learning_rate, max_depth, objective, X_train, y_train, y_train_pred = train_xgboost(train_data)

    # Evaluate the XGBoost classifier
    X_test = test_data.drop(columns=['fetal_health'])  # Features
    y_test = test_data['fetal_health']  # Target variable
    evaluate_classifier(xgb_clf, max_depth, n_estimators, X_train, y_train, y_train_pred, X_test, y_test)
