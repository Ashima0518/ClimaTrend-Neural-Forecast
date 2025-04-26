#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather data generation module for the ClimaTrend Neural Forecast system.
This module generates synthetic historical weather data for model training.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
RAW_WEATHER_DIR = os.path.join(DATA_DIR, 'raw', 'weather_data')
PROCESSED_WEATHER_DIR = os.path.join(DATA_DIR, 'processed', 'weather_data')

# Ensure directories exist
os.makedirs(RAW_WEATHER_DIR, exist_ok=True)
os.makedirs(PROCESSED_WEATHER_DIR, exist_ok=True)

# Weather parameters to generate
WEATHER_PARAMS = [
    'temp_c', 'humidity', 'wind_kph', 'wind_degree', 
    'pressure_mb', 'precip_mm', 'cloud'
]


def generate_historical_weather_data(locations, start_date, end_date):
    """
    Generate synthetic historical weather data for multiple locations within a date range.
    
    Args:
        locations (list): List of location dictionaries with 'name', 'lat', 'lon'
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        dict: Dictionary with location names as keys and weather dataframes as values
    """
    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate number of days
    delta = end_dt - start_dt
    num_days = delta.days + 1
    
    weather_data = {}
    
    for location in tqdm(locations, desc="Generating data for locations"):
        location_name = location['name']
        lat, lon = location['lat'], location['lon']
        
        # Generate all hourly data for the date range
        all_hours_data = []
        
        # Loop through each day
        current_dt = start_dt
        for day in tqdm(range(num_days), desc=f"Generating days for {location_name}"):
            # Seasonal factor (0-1) based on day of year (assumes northern hemisphere)
            day_of_year = current_dt.timetuple().tm_yday
            seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            # Latitude factor (0-1) - higher latitudes have more seasonal variation
            lat_factor = abs(lat) / 90.0
            
            # Baseline temperature based on latitude (warmer at equator)
            base_temp = 30 - 0.4 * abs(lat)
            
            # Add seasonal variation
            seasonal_temp_range = 25 * lat_factor
            base_temp += (seasonal_factor - 0.5) * seasonal_temp_range
            
            # Generate hourly data
            for hour in range(24):
                hour_dt = current_dt + timedelta(hours=hour)
                
                # Time of day factor (0-1) - warmest in afternoon, coolest before dawn
                hour_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (hour - 3) / 24)
                
                # Generate weather parameters with realistic patterns
                
                # Temperature: diurnal variation + random noise
                temp_c = base_temp + hour_factor * 8 - 4 + np.random.normal(0, 2)
                
                # Humidity: inverse correlation with temperature + random noise
                humidity = 70 - (temp_c - base_temp) * 3 + np.random.normal(0, 10)
                humidity = np.clip(humidity, 10, 100)
                
                # Wind: more variable
                wind_kph = 5 + 15 * np.random.beta(2, 5) + seasonal_factor * 10
                wind_degree = (day * 20 + hour * 5 + np.random.randint(-20, 20)) % 360
                
                # Pressure: slow variations around normal
                pressure_mb = 1013 + np.sin(day / 7 * np.pi) * 10 + np.random.normal(0, 1)
                
                # Precipitation: occasional showers, more frequent in certain seasons
                rain_chance = 0.1 + 0.3 * seasonal_factor * np.random.beta(2, 5)
                precip_mm = np.random.exponential(5) if np.random.random() < rain_chance else 0
                
                # Cloud cover: correlated with precipitation and humidity
                cloud_base = 20 + humidity * 0.5
                cloud = cloud_base + (40 if precip_mm > 0 else 0) + np.random.normal(0, 10)
                cloud = np.clip(cloud, 0, 100)
                
                # Create data entry
                hour_dict = {
                    'datetime': hour_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'temp_c': round(temp_c, 1),
                    'temp_f': round(temp_c * 9/5 + 32, 1),
                    'humidity': round(humidity),
                    'wind_kph': round(wind_kph, 1),
                    'wind_degree': round(wind_degree),
                    'pressure_mb': round(pressure_mb, 1),
                    'precip_mm': round(precip_mm, 1),
                    'cloud': round(cloud)
                }
                
                all_hours_data.append(hour_dict)
                
                # Save raw data to JSON file (one file per day)
                if hour == 0:
                    date_str = current_dt.strftime('%Y-%m-%d')
                    filename = f"{location_name.replace(' ', '_')}_{date_str}.json"
                    file_path = os.path.join(RAW_WEATHER_DIR, filename)
                    
                    # Create a structure similar to what an API might return
                    day_data = {
                        "location": {
                            "name": location_name,
                            "lat": lat,
                            "lon": lon
                        },
                        "forecast": {
                            "forecastday": [{
                                "date": date_str,
                                "hour": []
                            }]
                        }
                    }
                    
                    # We'll populate the hours as we generate them
                    day_data["forecast"]["forecastday"][0]["hour"] = []
            
            # Save the day's data
            date_str = current_dt.strftime('%Y-%m-%d')
            filename = f"{location_name.replace(' ', '_')}_{date_str}.json"
            file_path = os.path.join(RAW_WEATHER_DIR, filename)
            
            # Extract just this day's hours
            day_hours = [h for h in all_hours_data if h['datetime'].startswith(date_str)]
            
            # Create a structure similar to what an API might return
            day_data = {
                "location": {
                    "name": location_name,
                    "lat": lat,
                    "lon": lon
                },
                "forecast": {
                    "forecastday": [{
                        "date": date_str,
                        "hour": day_hours
                    }]
                }
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(day_data, f, indent=2)
            
            # Move to next day
            current_dt += timedelta(days=1)
        
        # Convert all hours data to DataFrame
        df = pd.DataFrame(all_hours_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort by datetime
        df.sort_values('datetime', inplace=True)
        
        weather_data[location_name] = df
    
    return weather_data


def save_processed_data(weather_data):
    """
    Save processed weather data to CSV files.
    
    Args:
        weather_data (dict): Dictionary with location names as keys and weather dataframes as values
    """
    for location_name, df in weather_data.items():
        filename = f"{location_name.replace(' ', '_')}_weather_data.csv"
        file_path = os.path.join(PROCESSED_WEATHER_DIR, filename)
        df.to_csv(file_path, index=False)
        print(f"Saved processed data for {location_name} to {file_path}")


def main():
    """Main function to generate synthetic weather data."""
    # Sample locations (major cities)
    locations = [
        {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
        {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
        {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298},
        {'name': 'Miami', 'lat': 25.7617, 'lon': -80.1918},
        {'name': 'Seattle', 'lat': 47.6062, 'lon': -122.3321}
    ]
    
    # Date range for historical data (last 3 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    print(f"Generating synthetic weather data from {start_date} to {end_date}")
    
    # Generate data
    weather_data = generate_historical_weather_data(locations, start_date, end_date)
    
    # Save processed data
    save_processed_data(weather_data)
    
    print("Weather data generation complete.")


if __name__ == "__main__":
    main() 