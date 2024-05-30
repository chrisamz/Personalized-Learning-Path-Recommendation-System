# evaluation.py

"""
Evaluation Module for Personalized Learning Path Recommendation System

This module contains functions for evaluating the performance of learning style classification,
recommendation algorithm, and adaptive learning system using appropriate metrics.

Techniques Used:
- Model Evaluation
- Recommendation System Evaluation

Metrics Used:
- Accuracy
- Precision
- Recall
- F1-score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Libraries/Tools:
- scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import joblib
import tensorflow as tf

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        pass

    def load_data(self, filepath):
        """
        Load test data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath)

    def load_model(self, model_filepath):
        """
        Load a trained model from a file.
        
        :param model_filepath: str, path to the saved model
        :return: model, loaded model
        """
        return joblib.load(model_filepath)

    def evaluate_classification(self, model, X_test, y_test):
        """
        Evaluate a classification model using accuracy, precision, recall, and F1-score.
        
        :param model: trained model
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        return metrics

    def evaluate_regression(self, model, X_test, y_test):
        """
        Evaluate a regression model using MAE and RMSE.
        
        :param model: trained model
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        y_pred = model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return metrics

    def evaluate_adaptive_system(self, model, scaler, X_test, y_test):
        """
        Evaluate an adaptive learning system using MAE and RMSE.
        
        :param model: trained model
        :param scaler: scaler used for preprocessing
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        X_test = scaler.transform(X_test)
        y_pred = model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return metrics

if __name__ == "__main__":
    test_data_filepath = 'data/processed/preprocessed_student_data_test.csv'
    target_column = 'learning_path'

    evaluator = ModelEvaluation()
    data = evaluator.load_data(test_data_filepath)
    X_test = data.drop(columns=[target_column])
    y_test = data[target_column]

    # Load and evaluate learning style classification model
    classification_model = evaluator.load_model('models/decision_tree_model.pkl')
    classification_metrics = evaluator.evaluate_classification(classification_model, X_test, y_test)
    print("Learning Style Classification Evaluation:", classification_metrics)

    # Load and evaluate recommendation model
    recommendation_model = evaluator.load_model('models/recommendation_model.pkl')
    recommendation_metrics = evaluator.evaluate_regression(recommendation_model, X_test, y_test)
    print("Recommendation Model Evaluation:", recommendation_metrics)

    # Load and evaluate adaptive learning system
    scaler = joblib.load('models/scaler.pkl')
    adaptive_model = tf.keras.models.load_model('models/adaptive_learning_model.h5')
    adaptive_metrics = evaluator.evaluate_adaptive_system(adaptive_model, scaler, X_test, y_test)
    print("Adaptive Learning System Evaluation:", adaptive_metrics)
