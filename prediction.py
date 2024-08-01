import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def predict_and_evaluate(model_file, X_test_file, y_test_file, algorithm_name):
    try:
        # Load the model from the pickle file
        model = joblib.load(model_file)
        
        # Load the test data
        X_test = pd.read_csv(X_test_file)
        y_test = pd.read_csv(y_test_file)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        
        # Print the evaluation metrics
        print(f"Evaluation for {algorithm_name}:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:\n", report)
        print("\nConfusion Matrix:\n", matrix)
        
        return accuracy, report, matrix
    except Exception as e:
        print("Error occurred while making predictions: ", str(e))
        return None, None, None

# Example usage
predict_and_evaluate("decision_tree_model.pkl", "X_test.csv", "y_test.csv", "Decision Tree")
