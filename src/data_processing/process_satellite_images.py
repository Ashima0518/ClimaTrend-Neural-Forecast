#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Satellite image generation module for the ClimaTrend Neural Forecast system.
This module generates synthetic satellite imagery for model training.
"""

import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
RAW_SATELLITE_DIR = os.path.join(DATA_DIR, 'raw', 'satellite_images')
PROCESSED_SATELLITE_DIR = os.path.join(DATA_DIR, 'processed', 'satellite_images')

# Ensure directories exist
os.makedirs(RAW_SATELLITE_DIR, exist_ok=True)
os.makedirs(PROCESSED_SATELLITE_DIR, exist_ok=True)


def generate_satellite_images(locations, start_date, end_date, interval_days=1):
    """
    Generate synthetic satellite images for specified locations and date range.
    
    Args:
        locations (list): List of location dictionaries with 'name', 'lat', 'lon'
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        interval_days (int): Interval between images in days
    
    Returns:
        dict: Dictionary with location names as keys and lists of image paths as values
    """
    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    image_paths = {}
    
    for location in tqdm(locations, desc="Generating images for locations"):
        location_name = location['name']
        lat, lon = location['lat'], location['lon']
        
        location_paths = []
        
        # Loop through each day with the specified interval
        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # Check if we already have an image for this location and date
            filename = f"{location_name.replace(' ', '_')}_{date_str}.png"
            file_path = os.path.join(RAW_SATELLITE_DIR, filename)
            
            if os.path.exists(file_path):
                # Image already exists
                location_paths.append(file_path)
            else:
                # Generate synthetic satellite image
                # We'll use the date and location to seed the random generator for consistency
                seed = int(current_dt.timestamp()) + hash(location_name) % 1000000
                np.random.seed(seed)
                
                # Generate image
                image = generate_synthetic_satellite_image(location, current_dt, (512, 512))
                
                # Save the image
                cv2.imwrite(file_path, image)
                location_paths.append(file_path)
                print(f"Generated satellite image for {location_name} on {date_str}")
            
            # Move to next interval
            current_dt += timedelta(days=interval_days)
        
        image_paths[location_name] = location_paths
    
    return image_paths


def generate_synthetic_satellite_image(location, date, image_size=(512, 512)):
    """
    Generate a synthetic satellite image for a specific location and date.
    
    Args:
        location (dict): Location dictionary with 'name', 'lat', 'lon'
        date (datetime): Date for the image
        image_size (tuple): Size of the generated image (width, height)
        
    Returns:
        np.array: Generated satellite image (BGR format for OpenCV)
    """
    # Get location information
    lat, lon = location['lat'], location['lon']
    
    # Create base image
    width, height = image_size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Determine color palette based on location (desert, forest, water, city)
    if abs(lat) < 30:  # Tropical/subtropical regions
        if -60 <= lon <= 40:  # Africa/Middle East
            base_color = (80, 120, 30)  # More desert/savanna
        else:
            base_color = (30, 100, 50)  # More tropical forest
    elif 30 <= abs(lat) <= 60:  # Temperate regions
        base_color = (50, 100, 70)  # More temperate forest/fields
    else:  # Polar regions
        base_color = (200, 200, 200)  # More snow/ice
    
    # Adjust for season
    season_factor = np.sin(2 * np.pi * (date.timetuple().tm_yday - 80) / 365)
    
    # Greener in summer for northern hemisphere, opposite for southern
    if lat > 0:  # Northern hemisphere
        green_adjust = int(30 * season_factor)
    else:  # Southern hemisphere
        green_adjust = int(-30 * season_factor)
    
    # Apply seasonal adjustment to base color
    base_color = (
        np.clip(base_color[0], 0, 255),
        np.clip(base_color[1] + green_adjust, 0, 255),
        np.clip(base_color[2], 0, 255)
    )
    
    # Fill base with slightly varied color
    for y in range(height):
        for x in range(width):
            noise = np.random.randint(-10, 10, 3)
            pixel_color = (
                np.clip(base_color[0] + noise[0], 0, 255),
                np.clip(base_color[1] + noise[1], 0, 255),
                np.clip(base_color[2] + noise[2], 0, 255)
            )
            img[y, x] = pixel_color
    
    # Create a PIL Image for easier drawing
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Add cloud patterns (white blobs) - varies by date
    cloud_seed = int(date.timestamp()) % 1000
    np.random.seed(cloud_seed)
    
    # Determine cloud coverage (0-1) based on location and date
    # More clouds in coastal and higher latitude areas
    base_cloud_coverage = 0.2 + 0.3 * (abs(lat) / 90)
    
    # Seasonal variation in cloud coverage
    seasonal_cloud_factor = 0.1 * np.sin(2 * np.pi * (date.timetuple().tm_yday - 80) / 365)
    cloud_coverage = base_cloud_coverage + seasonal_cloud_factor
    
    # Generate clouds
    num_clouds = int(20 * cloud_coverage)
    for _ in range(num_clouds):
        # Cloud center
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        
        # Cloud size
        cloud_size = np.random.randint(20, 100)
        
        # Cloud shape (ellipse)
        cloud_width = cloud_size * (0.5 + np.random.random())
        cloud_height = cloud_size * (0.5 + np.random.random())
        
        # Cloud opacity (0-255)
        opacity = np.random.randint(100, 230)
        
        # Draw cloud
        draw.ellipse(
            (cx - cloud_width, cy - cloud_height, cx + cloud_width, cy + cloud_height),
            fill=(255, 255, 255, opacity)
        )
    
    # Add some water bodies (blue)
    num_water = np.random.randint(0, 3)
    for _ in range(num_water):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        water_size = np.random.randint(30, 150)
        draw.ellipse(
            (cx - water_size, cy - water_size, cx + water_size, cy + water_size),
            fill=(20, 50, 150, 200)
        )
    
    # Add some roads/urban features (grayish lines)
    num_roads = np.random.randint(3, 10)
    for _ in range(num_roads):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(0, width)
        y2 = np.random.randint(0, height)
        road_width = np.random.randint(1, 5)
        draw.line((x1, y1, x2, y2), fill=(120, 120, 120), width=road_width)
    
    # Apply slight blur to simulate atmosphere
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # Convert back to OpenCV format (BGR)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    return img


def preprocess_satellite_images(image_paths, target_size=(224, 224)):
    """
    Preprocess satellite images for model input.
    
    Args:
        image_paths (dict): Dictionary with location names as keys and lists of image paths as values
        target_size (tuple): Target size for resizing images (width, height)
    
    Returns:
        dict: Dictionary with location names as keys and preprocessed image arrays as values
    """
    processed_images = {}
    
    for location_name, paths in image_paths.items():
        location_images = []
        
        for img_path in tqdm(paths, desc=f"Processing images for {location_name}"):
            try:
                # Read and process image
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                # Convert to RGB (OpenCV uses BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                
                # Apply image enhancements
                img_enhanced = enhance_satellite_image(img_resized)
                
                # Save processed image
                processed_filename = os.path.basename(img_path).replace('.png', '_processed.png')
                processed_path = os.path.join(PROCESSED_SATELLITE_DIR, processed_filename)
                cv2.imwrite(processed_path, cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR))
                
                # Normalize pixel values to [0, 1]
                img_normalized = img_enhanced.astype(np.float32) / 255.0
                
                location_images.append(img_normalized)
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
        
        if location_images:
            processed_images[location_name] = np.array(location_images)
    
    return processed_images


def enhance_satellite_image(img):
    """
    Enhance satellite image for better feature extraction.
    
    Args:
        img (np.array): Input image array
        
    Returns:
        np.array: Enhanced image
    """
    # Convert to float for processing
    img_float = img.astype(np.float32)
    
    # Apply contrast enhancement
    # Histogram equalization for better contrast
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # Apply slight sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    img_enhanced = cv2.filter2D(img_enhanced, -1, kernel)
    
    # Ensure values are within [0, 255]
    img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)
    
    return img_enhanced


def extract_image_features(processed_images):
    """
    Extract features from processed satellite images.
    
    Args:
        processed_images (dict): Dictionary with location names as keys and preprocessed image arrays
        
    Returns:
        dict: Dictionary with location names as keys and feature arrays as values
    """
    # For demonstration purposes, we're using a simple feature extraction technique
    # In a real-world scenario, you might use pre-trained CNNs like VGG, ResNet, etc.
    
    image_features = {}
    
    for location_name, images in processed_images.items():
        features = []
        
        for img in images:
            # Calculate basic statistical features
            channels_mean = np.mean(img, axis=(0, 1))
            channels_std = np.std(img, axis=(0, 1))
            channels_min = np.min(img, axis=(0, 1))
            channels_max = np.max(img, axis=(0, 1))
            
            # Calculate texture features using Haralick texture
            gray_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Simple edge detection features
            edges = cv2.Canny(gray_img, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Combine all features
            image_feature = np.concatenate([
                channels_mean, channels_std, channels_min, channels_max, [edge_density]
            ])
            
            features.append(image_feature)
        
        image_features[location_name] = np.array(features)
    
    return image_features


def save_image_features(image_features):
    """
    Save extracted image features to files.
    
    Args:
        image_features (dict): Dictionary with location names as keys and feature arrays as values
    """
    for location_name, features in image_features.items():
        filename = f"{location_name.replace(' ', '_')}_image_features.npy"
        file_path = os.path.join(PROCESSED_SATELLITE_DIR, filename)
        np.save(file_path, features)
        print(f"Saved image features for {location_name} to {file_path}")


def main():
    """Main function to generate and process synthetic satellite images."""
    # Sample locations (same as in fetch_weather_data.py)
    locations = [
        {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
        {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
        {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298},
        {'name': 'Miami', 'lat': 25.7617, 'lon': -80.1918},
        {'name': 'Seattle', 'lat': 47.6062, 'lon': -122.3321}
    ]
    
    # Date range for satellite images (last year, with weekly intervals)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Generating synthetic satellite images from {start_date} to {end_date}")
    
    # Generate images
    image_paths = generate_satellite_images(locations, start_date, end_date, interval_days=7)
    
    # Preprocess images
    processed_images = preprocess_satellite_images(image_paths)
    
    # Extract features
    image_features = extract_image_features(processed_images)
    
    # Save features
    save_image_features(image_features)
    
    print("Satellite image generation and processing complete.")


if __name__ == "__main__":
    main() 