import pandas as pd
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

warnings.filterwarnings('ignore')

def load_data(file_name):
    try:
        data = pd.read_csv(file_name)
        return data
    except Exception as e:
        print(f"Error occurred while loading the data from {file_name}: ", str(e))

def train_bagging_classifier(train_data):
    try:
        X_train = train_data.drop(columns=['fetal_health'])  # Features
        y_train = train_data['fetal_health']  # fetal_health variable

        # Define parameters for the Bagging Classifier
        criterion = 'gini'
        max_depth = None
        n_estimators = 10
    
        # Initialize the base Decision Tree classifier
        base_clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
        
        # Initialize and train the Bagging classifier
        bagging_clf = BaggingClassifier(estimator=base_clf, n_estimators=n_estimators, random_state=42)
        bagging_clf.fit(X_train, y_train)

        # Make predictions on the training data
        y_train_pred = bagging_clf.predict(X_train)
        
        return bagging_clf, criterion, max_depth, n_estimators, X_train, y_train, y_train_pred
    except Exception as e:
        print("Error occurred while training the bagging classifier: ", str(e))

def evaluate_classifier(clf, criterion, max_depth, n_estimators, X_train, y_train, y_train_pred, X_test, y_test):
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
        
        # Output the criterion, max depth, number of estimators, and performance
        print(f"CRITERION: {criterion}")
        print(f"MAX_DEPTH: {max_depth}")
        print(f"N_ESTIMATORS: {n_estimators}")
        print(f"TRAINING ACCURACY: {train_accuracy * 100:.2f}%")
        print(f"TESTING ACCURACY: {test_accuracy * 100:.2f}%")
        print("\nTraining Classification Report:\n", train_report)
        print("\nTesting Classification Report:\n", test_report)
        print("\nTraining Confusion Matrix:\n", train_matrix)
        print("\nTesting Confusion Matrix:\n", test_matrix)
        
        # Save the trained model to a file
        model_filename = 'bagging_decision_tree_model.pkl'
        joblib.dump(clf, model_filename)
        print(f"Trained model saved as {model_filename}")
    except Exception as e:
        print("Error occurred while evaluating the bagging classifier: ", str(e))

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

    # Train the Bagging classifier
    clf, criterion, max_depth, n_estimators, X_train, y_train, y_train_pred = train_bagging_classifier(train_data)
    
    # Evaluate the Bagging classifier
    X_test = test_data.drop(columns=['fetal_health'])  # Features
    y_test = test_data['fetal_health']  # fetal_health variable
    evaluate_classifier(clf, criterion, max_depth, n_estimators, X_train, y_train, y_train_pred, X_test, y_test)
