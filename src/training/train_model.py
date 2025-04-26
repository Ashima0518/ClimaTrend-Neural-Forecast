#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model training module for the ClimaTrend Neural Forecast system.
This module handles training and fine-tuning of the weather prediction models.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our models
from models.weather_cnn import (
    create_simple_cnn, 
    create_resnet_weather_model,
    create_efficient_weather_model,
    create_lstm_weather_model,
    create_hybrid_model
)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Weather parameters to predict
WEATHER_PARAMS = [
    'temp_c', 'humidity', 'wind_kph', 'wind_degree', 
    'pressure_mb', 'precip_mm', 'cloud'
]


def load_and_prepare_data():
    """
    Load and prepare data for model training.
    
    Returns:
        tuple: (X_image, X_time_series, y) data arrays for training
    """
    print("Loading and preparing data...")
    
    # Load processed weather data
    weather_data_dir = os.path.join(DATA_DIR, 'processed', 'weather_data')
    satellite_data_dir = os.path.join(DATA_DIR, 'processed', 'satellite_images')
    
    all_weather_df = []
    
    # Load all weather CSV files
    for filename in os.listdir(weather_data_dir):
        if filename.endswith('_weather_data.csv'):
            file_path = os.path.join(weather_data_dir, filename)
            df = pd.read_csv(file_path)
            location_name = filename.replace('_weather_data.csv', '')
            df['location'] = location_name
            all_weather_df.append(df)
    
    # Combine all weather data
    if all_weather_df:
        weather_df = pd.concat(all_weather_df, ignore_index=True)
        print(f"Loaded weather data with {weather_df.shape[0]} records")
    else:
        raise ValueError("No weather data files found")
    
    # Load satellite image features
    image_features = {}
    for filename in os.listdir(satellite_data_dir):
        if filename.endswith('_image_features.npy'):
            file_path = os.path.join(satellite_data_dir, filename)
            location_name = filename.replace('_image_features.npy', '')
            features = np.load(file_path)
            image_features[location_name] = features
    
    # Check if we have image features
    if not image_features:
        raise ValueError("No image feature files found")
    
    # Prepare X and y data
    X_images = []
    X_time_series = []
    y_values = []
    
    # Group by location
    for location, group in weather_df.groupby('location'):
        # Skip if we don't have image features for this location
        if location not in image_features:
            continue
        
        # Sort by datetime
        group = group.sort_values('datetime')
        
        # Extract features and target values
        location_features = image_features[location]
        
        # Only use rows where we have images (assuming 1:1 correspondence)
        if len(group) >= len(location_features):
            group = group.iloc[:len(location_features)]
            
            # Prepare time series data (previous 24 hours)
            for i in range(24, len(group)):
                # Image feature for the current time
                X_images.append(location_features[i])
                
                # Time series data (previous 24 hours)
                time_series = group.iloc[i-24:i][WEATHER_PARAMS].values
                X_time_series.append(time_series)
                
                # Target values (current hour)
                y_values.append(group.iloc[i][WEATHER_PARAMS].values)
    
    # Convert to numpy arrays
    X_images = np.array(X_images)
    X_time_series = np.array(X_time_series)
    y_values = np.array(y_values)
    
    print(f"Prepared data: {len(X_images)} samples")
    print(f"X_images shape: {X_images.shape}")
    print(f"X_time_series shape: {X_time_series.shape}")
    print(f"y_values shape: {y_values.shape}")
    
    return X_images, X_time_series, y_values


def train_test_split(X_images, X_time_series, y, test_size=0.2, val_size=0.1):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X_images (np.array): Image features
        X_time_series (np.array): Time series features
        y (np.array): Target values
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        
    Returns:
        tuple: Training, validation, and test data
    """
    # Shuffle the data
    indices = np.arange(len(X_images))
    np.random.shuffle(indices)
    
    X_images = X_images[indices]
    X_time_series = X_time_series[indices]
    y = y[indices]
    
    # Split into train+val and test
    test_count = int(len(X_images) * test_size)
    
    X_test_images = X_images[:test_count]
    X_test_time_series = X_time_series[:test_count]
    y_test = y[:test_count]
    
    X_train_val_images = X_images[test_count:]
    X_train_val_time_series = X_time_series[test_count:]
    y_train_val = y[test_count:]
    
    # Split train+val into train and val
    val_count = int(len(X_train_val_images) * val_size)
    
    X_val_images = X_train_val_images[:val_count]
    X_val_time_series = X_train_val_time_series[:val_count]
    y_val = y_train_val[:val_count]
    
    X_train_images = X_train_val_images[val_count:]
    X_train_time_series = X_train_val_time_series[val_count:]
    y_train = y_train_val[val_count:]
    
    print(f"Training samples: {len(X_train_images)}")
    print(f"Validation samples: {len(X_val_images)}")
    print(f"Test samples: {len(X_test_images)}")
    
    return (
        (X_train_images, X_train_time_series, y_train),
        (X_val_images, X_val_time_series, y_val),
        (X_test_images, X_test_time_series, y_test)
    )


def setup_callbacks(model_name):
    """
    Set up callbacks for model training.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        list: List of callbacks
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.h5")
    log_dir = os.path.join(LOGS_DIR, f"{model_name}_{timestamp}")
    
    callbacks = [
        # Save the best model
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=1,
            restore_best_weights=True
        ),
        
        # Reduce learning rate when a metric has stopped improving
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks


def train_simple_cnn(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train a simple CNN model for weather prediction.
    
    Args:
        X_train (np.array): Training images
        y_train (np.array): Training targets
        X_val (np.array): Validation images
        y_val (np.array): Validation targets
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        
    Returns:
        tf.keras.Model: Trained model
    """
    print("Training simple CNN model...")
    
    # Create model
    model = create_simple_cnn(input_shape=X_train.shape[1:], output_dims=y_train.shape[1])
    
    # Print model summary
    model.summary()
    
    # Set up callbacks
    callbacks = setup_callbacks('simple_cnn')
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, 'simple_cnn')
    
    return model


def train_resnet_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train a ResNet-based model for weather prediction.
    
    Args:
        X_train (np.array): Training images
        y_train (np.array): Training targets
        X_val (np.array): Validation images
        y_val (np.array): Validation targets
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        
    Returns:
        tf.keras.Model: Trained model
    """
    print("Training ResNet-based model...")
    
    # Create model
    model = create_resnet_weather_model(
        input_shape=X_train.shape[1:], 
        output_dims=y_train.shape[1],
        freeze_base=True
    )
    
    # Print model summary
    model.summary()
    
    # Set up callbacks
    callbacks = setup_callbacks('resnet_weather')
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, 'resnet_weather')
    
    # Fine-tune the model by unfreezing some layers
    print("Fine-tuning ResNet model...")
    
    # Unfreeze the top layers of the base model
    for layer in model.layers[0].layers[-30:]:
        layer.trainable = True
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='mse',
        metrics=['mae']
    )
    
    # Set up callbacks for fine-tuning
    callbacks = setup_callbacks('resnet_weather_finetuned')
    
    # Train again with a lower learning rate
    history_ft = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs // 2,  # Fewer epochs for fine-tuning
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot fine-tuning history
    plot_training_history(history_ft, 'resnet_weather_finetuned')
    
    return model


def train_hybrid_model(X_train_images, X_train_time_series, y_train, 
                       X_val_images, X_val_time_series, y_val,
                       epochs=50, batch_size=32):
    """
    Train a hybrid model combining CNN and LSTM for weather prediction.
    
    Args:
        X_train_images (np.array): Training images
        X_train_time_series (np.array): Training time series
        y_train (np.array): Training targets
        X_val_images (np.array): Validation images
        X_val_time_series (np.array): Validation time series
        y_val (np.array): Validation targets
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        
    Returns:
        tf.keras.Model: Trained model
    """
    print("Training hybrid CNN-LSTM model...")
    
    # Create model
    model = create_hybrid_model(
        image_input_shape=X_train_images.shape[1:],
        time_series_shape=X_train_time_series.shape[1:],
        output_dims=y_train.shape[1]
    )
    
    # Print model summary
    model.summary()
    
    # Set up callbacks
    callbacks = setup_callbacks('hybrid_cnn_lstm')
    
    # Train the model
    history = model.fit(
        {'image_input': X_train_images, 'time_series_input': X_train_time_series},
        y_train,
        validation_data=(
            {'image_input': X_val_images, 'time_series_input': X_val_time_series},
            y_val
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, 'hybrid_cnn_lstm')
    
    return model


def plot_training_history(history, model_name):
    """
    Plot training history and save the figure.
    
    Args:
        history (tf.keras.callbacks.History): Training history
        model_name (str): Name of the model
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation mean absolute error
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Save the figure
    plt.tight_layout()
    plot_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{model_name}_history.png'))
    plt.close()


def main():
    """Main function to train weather prediction models."""
    print("ClimaTrend Neural Forecast - Model Training")
    
    try:
        # Load and prepare data
        X_images, X_time_series, y = load_and_prepare_data()
        
        # Split data
        (X_train_images, X_train_time_series, y_train), \
        (X_val_images, X_val_time_series, y_val), \
        (X_test_images, X_test_time_series, y_test) = train_test_split(
            X_images, X_time_series, y
        )
        
        # Set number of epochs and batch size
        epochs = 50
        batch_size = 32
        
        # Train models
        models = {}
        
        # 1. Simple CNN model
        models['simple_cnn'] = train_simple_cnn(
            X_train_images, y_train,
            X_val_images, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # 2. ResNet-based model
        models['resnet'] = train_resnet_model(
            X_train_images, y_train,
            X_val_images, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # 3. Hybrid CNN-LSTM model
        models['hybrid'] = train_hybrid_model(
            X_train_images, X_train_time_series, y_train,
            X_val_images, X_val_time_series, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print("Model training complete.")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 