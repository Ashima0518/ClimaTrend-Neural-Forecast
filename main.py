#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module for the ClimaTrend Neural Forecast system.
This script runs the complete weather forecasting pipeline.
"""

import os
import argparse
import time
from datetime import datetime

# Import modules
from src.data_processing import fetch_weather_data, process_satellite_images
from src.training import train_model
from src.evaluation import evaluate_models
from src.visualization import visualize_weather


def setup_argparse():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(description='ClimaTrend Neural Forecast System')
    
    parser.add_argument('--fetch-data', action='store_true',
                       help='Generate synthetic weather data')
    
    parser.add_argument('--process-images', action='store_true',
                       help='Generate and process synthetic satellite images')
    
    parser.add_argument('--train', action='store_true',
                       help='Train weather prediction models')
    
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained models')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Create weather forecast visualizations')
    
    parser.add_argument('--all', action='store_true',
                       help='Run the complete pipeline')
    
    return parser.parse_args()


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def main():
    """Main function to run the weather forecasting pipeline."""
    print_header("ClimaTrend Neural Forecast System")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse command-line arguments
    args = setup_argparse()
    
    # If --all is specified, run the complete pipeline
    if args.all:
        args.fetch_data = True
        args.process_images = True
        args.train = True
        args.evaluate = True
        args.visualize = True
    
    # Track timing
    start_time = time.time()
    
    # Step 1: Generate synthetic weather data
    if args.fetch_data:
        print_header("Step 1: Generating Synthetic Weather Data")
        fetch_weather_data.main()
    
    # Step 2: Generate and process synthetic satellite images
    if args.process_images:
        print_header("Step 2: Generating and Processing Synthetic Satellite Images")
        process_satellite_images.main()
    
    # Step 3: Train models
    if args.train:
        print_header("Step 3: Training Weather Prediction Models")
        train_model.main()
    
    # Step 4: Evaluate models
    if args.evaluate:
        print_header("Step 4: Evaluating Models")
        evaluate_models.main()
    
    # Step 5: Visualize weather forecasts
    if args.visualize:
        print_header("Step 5: Creating Weather Visualizations")
        visualize_weather.main()
    
    # Print completion message
    elapsed_time = time.time() - start_time
    print_header("Pipeline Completed")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main() 