#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model evaluation module for the ClimaTrend Neural Forecast system.
This module handles evaluation of trained weather prediction models.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Weather parameters to predict
WEATHER_PARAMS = [
    'temp_c', 'humidity', 'wind_kph', 'wind_degree', 
    'pressure_mb', 'precip_mm', 'cloud'
]


def find_best_model(model_type):
    """
    Find the best model of a given type based on the filename pattern.
    
    Args:
        model_type (str): Type of model to find ('simple_cnn', 'resnet_weather', etc.)
        
    Returns:
        str: Path to the best model file
    """
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(model_type) and f.endswith('.h5')]
    
    if not model_files:
        raise ValueError(f"No {model_type} models found in {MODELS_DIR}")
    
    # Return the most recent model (assuming timestamp format in filename)
    return os.path.join(MODELS_DIR, sorted(model_files)[-1])


def load_test_data():
    """
    Load test data for model evaluation.
    This is a simplified version; in a real scenario, you would load your actual test data.
    
    Returns:
        tuple: Test data (X_images, X_time_series, y)
    """
    print("Loading test data...")
    
    # This is a placeholder. In a real-world scenario, you would:
    # 1. Load your preprocessed test data
    # 2. Or run the data preparation code from train_model.py with test=True
    
    # For demonstration, we'll create a dummy test set
    # In practice, replace this with actual test data loading
    
    # Attempt to load cached test data if it exists
    test_data_path = os.path.join(DATA_DIR, 'processed', 'test_data.npz')
    if os.path.exists(test_data_path):
        print(f"Loading cached test data from {test_data_path}")
        test_data = np.load(test_data_path, allow_pickle=True)
        return test_data['X_images'], test_data['X_time_series'], test_data['y']
    
    # If no cached data, create dummy data
    # This is only for demonstration - replace with actual data loading!
    print("Warning: Using dummy test data. Replace with actual test data loading!")
    
    # Image features (13 features per image)
    X_images = np.random.rand(100, 13)
    
    # Time series data (24 hours, 7 weather parameters)
    X_time_series = np.random.rand(100, 24, len(WEATHER_PARAMS))
    
    # Target values (7 weather parameters)
    y = np.random.rand(100, len(WEATHER_PARAMS))
    
    print(f"Loaded test data: {len(X_images)} samples")
    
    return X_images, X_time_series, y


def predict_with_model(model, X_images, X_time_series=None, is_hybrid=False):
    """
    Make predictions with a trained model.
    
    Args:
        model (tf.keras.Model): Trained model
        X_images (np.array): Image features
        X_time_series (np.array): Time series features (for hybrid models)
        is_hybrid (bool): Whether the model is a hybrid CNN-LSTM model
        
    Returns:
        np.array: Predicted values
    """
    if is_hybrid:
        predictions = model.predict({
            'image_input': X_images,
            'time_series_input': X_time_series
        })
    else:
        predictions = model.predict(X_images)
    
    return predictions


def evaluate_model(model, X_images, X_time_series, y_true, is_hybrid=False):
    """
    Evaluate a trained model on test data.
    
    Args:
        model (tf.keras.Model): Trained model
        X_images (np.array): Image features
        X_time_series (np.array): Time series features (for hybrid models)
        y_true (np.array): True target values
        is_hybrid (bool): Whether the model is a hybrid CNN-LSTM model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = predict_with_model(model, X_images, X_time_series, is_hybrid)
    
    # Calculate metrics for each weather parameter
    metrics = {}
    
    for i, param in enumerate(WEATHER_PARAMS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        metrics[param] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        }
    
    # Calculate overall metrics
    metrics['overall'] = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    
    return metrics


def plot_predictions(y_true, y_pred, param_index, param_name, model_name):
    """
    Plot true vs predicted values for a specific weather parameter.
    
    Args:
        y_true (np.array): True target values
        y_pred (np.array): Predicted values
        param_index (int): Index of the parameter to plot
        param_name (str): Name of the parameter to plot
        model_name (str): Name of the model
    """
    plt.figure(figsize=(10, 6))
    
    # Get values for the specific parameter
    true_values = y_true[:, param_index]
    pred_values = y_pred[:, param_index]
    
    # Plot true values
    plt.plot(true_values, label='True Values', marker='o', linestyle='-', alpha=0.7)
    
    # Plot predicted values
    plt.plot(pred_values, label='Predicted Values', marker='x', linestyle='--', alpha=0.7)
    
    # Add labels and legend
    plt.title(f'{param_name} - True vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel(param_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_{param_name}_predictions.png'))
    plt.close()


def plot_error_distribution(y_true, y_pred, param_index, param_name, model_name):
    """
    Plot error distribution for a specific weather parameter.
    
    Args:
        y_true (np.array): True target values
        y_pred (np.array): Predicted values
        param_index (int): Index of the parameter to plot
        param_name (str): Name of the parameter to plot
        model_name (str): Name of the model
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate errors
    errors = y_true[:, param_index] - y_pred[:, param_index]
    
    # Plot histogram of errors
    plt.hist(errors, bins=30, alpha=0.7, color='blue')
    
    # Add labels
    plt.title(f'Error Distribution for {param_name}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at zero
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_{param_name}_error_distribution.png'))
    plt.close()


def plot_scatter(y_true, y_pred, param_index, param_name, model_name):
    """
    Plot scatter plot of true vs predicted values for a specific weather parameter.
    
    Args:
        y_true (np.array): True target values
        y_pred (np.array): Predicted values
        param_index (int): Index of the parameter to plot
        param_name (str): Name of the parameter to plot
        model_name (str): Name of the model
    """
    plt.figure(figsize=(10, 6))
    
    # Get values for the specific parameter
    true_values = y_true[:, param_index]
    pred_values = y_pred[:, param_index]
    
    # Plot scatter plot
    plt.scatter(true_values, pred_values, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(np.min(true_values), np.min(pred_values))
    max_val = max(np.max(true_values), np.max(pred_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels
    plt.title(f'{param_name} - True vs Predicted')
    plt.xlabel(f'True {param_name}')
    plt.ylabel(f'Predicted {param_name}')
    plt.grid(True, alpha=0.3)
    
    # Calculate R²
    r2 = r2_score(true_values, pred_values)
    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_{param_name}_scatter.png'))
    plt.close()


def evaluate_and_visualize_model(model, model_name, X_images, X_time_series, y_true, is_hybrid=False):
    """
    Evaluate a model and create visualizations.
    
    Args:
        model (tf.keras.Model): Trained model
        model_name (str): Name of the model
        X_images (np.array): Image features
        X_time_series (np.array): Time series features (for hybrid models)
        y_true (np.array): True target values
        is_hybrid (bool): Whether the model is a hybrid CNN-LSTM model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print(f"Evaluating {model_name} model...")
    
    # Make predictions
    y_pred = predict_with_model(model, X_images, X_time_series, is_hybrid)
    
    # Calculate metrics
    metrics = evaluate_model(model, X_images, X_time_series, y_true, is_hybrid)
    
    # Print metrics
    print(f"\nEvaluation metrics for {model_name}:")
    print("Overall Metrics:")
    for metric, value in metrics['overall'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nMetrics by Weather Parameter:")
    for param, param_metrics in metrics.items():
        if param != 'overall':
            print(f"  {param}:")
            for metric, value in param_metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    # Create visualizations for each weather parameter
    for i, param in enumerate(WEATHER_PARAMS):
        # Plot true vs predicted values
        plot_predictions(y_true, y_pred, i, param, model_name)
        
        # Plot error distribution
        plot_error_distribution(y_true, y_pred, i, param, model_name)
        
        # Plot scatter plot
        plot_scatter(y_true, y_pred, i, param, model_name)
    
    # Save metrics to JSON
    metrics_file = os.path.join(PLOTS_DIR, f'{model_name}_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    """Main function to evaluate weather prediction models."""
    print("ClimaTrend Neural Forecast - Model Evaluation")
    
    try:
        # Load test data
        X_images, X_time_series, y = load_test_data()
        
        # List of models to evaluate
        models_to_evaluate = [
            {'name': 'simple_cnn', 'is_hybrid': False},
            {'name': 'resnet_weather', 'is_hybrid': False},
            {'name': 'hybrid_cnn_lstm', 'is_hybrid': True}
        ]
        
        all_metrics = {}
        
        # Evaluate each model
        for model_info in models_to_evaluate:
            model_name = model_info['name']
            is_hybrid = model_info['is_hybrid']
            
            try:
                # Find and load the best model
                model_path = find_best_model(model_name)
                model = load_model(model_path)
                
                print(f"Loaded model from {model_path}")
                
                # Evaluate model
                metrics = evaluate_and_visualize_model(
                    model, model_name, X_images, X_time_series, y, is_hybrid
                )
                
                all_metrics[model_name] = metrics
                
            except Exception as e:
                print(f"Error evaluating {model_name} model: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Compare models
        if all_metrics:
            print("\nModel Comparison:")
            print("Overall MAE for each model:")
            
            for model_name, metrics in all_metrics.items():
                mae = metrics['overall']['MAE']
                print(f"  {model_name}: {mae:.4f}")
        
        print("\nModel evaluation complete.")
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 