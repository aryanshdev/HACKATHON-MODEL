import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

def tpot_automl(file_path, output_pipeline_filename):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Assume the target column is named 'fetal_health' (change it if necessary)
    target = 'fetal_health'
    X = data.drop(target, axis=1)
    y = data[target]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

    # Initialize the TPOT classifier with 5 generations and a population size of 50
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)

    # Fit the TPOT classifier on the training data
    tpot.fit(X_train, y_train)

    # Evaluate the classifier on the testing data
    y_pred = tpot.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Export the best pipeline
    tpot.export(output_pipeline_filename)

    # Extract the accuracy scores from TPOT's generations
    accuracy_scores = [t['internal_cv_score'] for t in tpot.evaluated_individuals_.values()]

    # Convert to numpy array and calculate z-score
    accuracy_scores = np.array(accuracy_scores)
    z_scores = stats.zscore(accuracy_scores)
    z_score = (accuracy - np.mean(accuracy_scores)) / np.std(accuracy_scores)
    area_to_left = stats.norm.cdf(z_score)
    p_value = (1 - area_to_left) * 2
    alpha = 0.05
    null_hypothesis_true = p_value > alpha

    print(f'Z-score for accuracy: {z_score}')
    print(f'P-value for accuracy: {p_value}')
    print(f'Null hypothesis is true: {null_hypothesis_true}')

    # Plot the distribution
    sns.histplot(accuracy_scores, kde=True, stat="density", bins=10, color="skyblue", label="Accuracy")
    x = np.linspace(min(accuracy_scores), max(accuracy_scores), 100)
    plt.plot(x, stats.norm.pdf(x, np.mean(accuracy_scores), np.std(accuracy_scores)), label="Normal Distribution")

    # Highlight the area to the left of the Z-score
    z_value_data_point = np.mean(accuracy_scores) + z_score * np.std(accuracy_scores)
    plt.fill_between(x, 0, stats.norm.pdf(x, np.mean(accuracy_scores), np.std(accuracy_scores)), where=(x <= z_value_data_point), color='gray', alpha=0.5)

    # Adding labels and legend
    plt.axvline(z_value_data_point, color='red', linestyle='--', label=f'Z-score {z_score}')
    plt.title('Accuracy Distribution with Normal Curve')
    plt.xlabel('Accuracy Values')
    plt.ylabel('Density')
    plt.legend()

    # Showing the plot
    plt.show()

# Example usage:
file_path = 'fetal_health.csv'
output_pipeline_filename = 'desired_pipeline_filename.py'
tpot_automl(file_path, output_pipeline_filename)
