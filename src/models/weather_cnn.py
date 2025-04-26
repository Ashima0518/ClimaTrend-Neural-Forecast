#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN model architecture for the ClimaTrend Neural Forecast system.
This module defines the deep learning models used for weather prediction.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0

def create_simple_cnn(input_shape=(224, 224, 3), output_dims=12):
    """
    Create a simple CNN model for weather prediction from satellite images.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        output_dims (int): Number of output dimensions (weather parameters to predict)
        
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(output_dims)
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_resnet_weather_model(input_shape=(224, 224, 3), output_dims=12, freeze_base=True):
    """
    Create a ResNet-based model for weather prediction from satellite images.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        output_dims (int): Number of output dimensions (weather parameters to predict)
        freeze_base (bool): Whether to freeze the base ResNet model
        
    Returns:
        tf.keras.Model: Compiled ResNet-based model
    """
    # Load the pre-trained ResNet model without the top classification layer
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model if requested
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Add custom top layers for weather prediction
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(output_dims)(x)
    
    # Create the complete model
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_efficient_weather_model(input_shape=(224, 224, 3), output_dims=12, freeze_base=True):
    """
    Create an EfficientNet-based model for weather prediction from satellite images.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        output_dims (int): Number of output dimensions (weather parameters to predict)
        freeze_base (bool): Whether to freeze the base EfficientNet model
        
    Returns:
        tf.keras.Model: Compiled EfficientNet-based model
    """
    # Load the pre-trained EfficientNet model without the top classification layer
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model if requested
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Add custom top layers for weather prediction
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(output_dims)(x)
    
    # Create the complete model
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_lstm_weather_model(timesteps, features, output_dims=12):
    """
    Create an LSTM model for time-series weather prediction.
    
    Args:
        timesteps (int): Number of timesteps in the input sequence
        features (int): Number of features per timestep
        output_dims (int): Number of output dimensions (weather parameters to predict)
        
    Returns:
        tf.keras.Model: Compiled LSTM model
    """
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(output_dims)
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_hybrid_model(image_input_shape=(224, 224, 3), 
                        time_series_shape=(24, 10), 
                        output_dims=12):
    """
    Create a hybrid model that combines CNN and LSTM for weather prediction.
    
    Args:
        image_input_shape (tuple): Input shape for satellite images
        time_series_shape (tuple): Input shape for time series data (timesteps, features)
        output_dims (int): Number of output dimensions (weather parameters to predict)
        
    Returns:
        tf.keras.Model: Compiled hybrid model
    """
    # Image input branch (CNN)
    image_input = layers.Input(shape=image_input_shape, name='image_input')
    
    # Use a lightweight CNN for image processing
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    image_features = layers.Dense(128, activation='relu')(x)
    
    # Time series input branch (LSTM)
    time_series_input = layers.Input(shape=time_series_shape, name='time_series_input')
    
    y = layers.LSTM(64, return_sequences=True)(time_series_input)
    y = layers.Dropout(0.2)(y)
    y = layers.LSTM(32)(y)
    time_series_features = layers.Dense(64, activation='relu')(y)
    
    # Combine both branches
    combined = layers.concatenate([image_features, time_series_features])
    
    # Final prediction layers
    z = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)
    output = layers.Dense(output_dims)(z)
    
    # Create the model with multiple inputs
    model = models.Model(
        inputs=[image_input, time_series_input],
        outputs=output
    )
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    return model 