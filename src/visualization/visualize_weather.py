#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather visualization module for the ClimaTrend Neural Forecast system.
This module creates visualizations of weather forecasts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Weather parameters to visualize
WEATHER_PARAMS = [
    'temp_c', 'humidity', 'wind_kph', 'wind_degree', 
    'pressure_mb', 'precip_mm', 'cloud'
]

# Parameter display names and units
PARAM_INFO = {
    'temp_c': {'name': 'Temperature', 'unit': '°C', 'cmap': 'RdYlBu_r'},
    'humidity': {'name': 'Humidity', 'unit': '%', 'cmap': 'Blues'},
    'wind_kph': {'name': 'Wind Speed', 'unit': 'kph', 'cmap': 'YlGnBu'},
    'wind_degree': {'name': 'Wind Direction', 'unit': '°', 'cmap': 'hsv'},
    'pressure_mb': {'name': 'Pressure', 'unit': 'mb', 'cmap': 'RdBu_r'},
    'precip_mm': {'name': 'Precipitation', 'unit': 'mm', 'cmap': 'Blues'},
    'cloud': {'name': 'Cloud Coverage', 'unit': '%', 'cmap': 'Greys'}
}


def load_forecast_data(location_name=None):
    """
    Load forecast data for visualization.
    
    Args:
        location_name (str, optional): Name of location to load data for.
                                      If None, loads data for all locations.
    
    Returns:
        pd.DataFrame: Dataframe with forecast data
    """
    # This is a placeholder. In a real scenario, you would:
    # 1. Load actual forecast data from your model outputs
    # 2. Or generate sample data for demonstration
    
    # For demonstration, let's create sample forecast data
    print("Creating sample forecast data for visualization...")
    
    # Sample locations
    locations = [
        {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
        {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
        {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298},
        {'name': 'Miami', 'lat': 25.7617, 'lon': -80.1918},
        {'name': 'Seattle', 'lat': 47.6062, 'lon': -122.3321}
    ]
    
    # Filter to specific location if provided
    if location_name:
        locations = [loc for loc in locations if loc['name'] == location_name]
        if not locations:
            raise ValueError(f"Location '{location_name}' not found")
    
    # Generate forecast data for next 7 days
    now = datetime.now()
    forecast_data = []
    
    for location in locations:
        # Create hourly forecasts for 7 days
        for day in range(7):
            for hour in range(24):
                forecast_time = now + timedelta(days=day, hours=hour)
                
                # Generate simulated weather parameters (with some temporal correlation)
                day_progress = day / 7
                hour_progress = hour / 24
                daily_cycle = np.sin(hour_progress * 2 * np.pi)
                
                # Temperature: cooler at night, warmer during day, with some daily variation
                temp_base = 20 + day_progress * 5  # Gradual warming trend
                temp_c = temp_base + 8 * daily_cycle + np.random.normal(0, 2)
                
                # Humidity: inverse of temperature cycle
                humidity = 60 - 20 * daily_cycle + np.random.normal(0, 5)
                humidity = np.clip(humidity, 0, 100)
                
                # Wind: more variable
                wind_kph = 10 + 5 * np.sin(day_progress * np.pi) + np.random.normal(0, 3)
                wind_kph = max(0, wind_kph)
                wind_degree = (day * 45 + hour * 5) % 360
                
                # Pressure: slow variations
                pressure_mb = 1013 + 10 * np.sin(day_progress * np.pi) + np.random.normal(0, 1)
                
                # Precipitation: occasional showers
                precip_prob = 0.2 + 0.3 * (np.sin(day_progress * 2 * np.pi) > 0.5)
                precip_mm = np.random.exponential(2) if np.random.random() < precip_prob else 0
                
                # Cloud cover: correlated with precipitation
                cloud = 20 + 60 * precip_prob + np.random.normal(0, 10)
                cloud = np.clip(cloud, 0, 100)
                
                # Create forecast entry
                forecast_entry = {
                    'location': location['name'],
                    'lat': location['lat'],
                    'lon': location['lon'],
                    'datetime': forecast_time,
                    'temp_c': temp_c,
                    'humidity': humidity,
                    'wind_kph': wind_kph,
                    'wind_degree': wind_degree,
                    'pressure_mb': pressure_mb,
                    'precip_mm': precip_mm,
                    'cloud': cloud
                }
                
                forecast_data.append(forecast_entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(forecast_data)
    
    print(f"Created sample forecast data with {len(df)} entries")
    
    return df


def plot_time_series_forecast(forecast_df, location_name, param):
    """
    Plot time series forecast for a specific location and weather parameter.
    
    Args:
        forecast_df (pd.DataFrame): Dataframe with forecast data
        location_name (str): Name of location to plot
        param (str): Weather parameter to plot
    """
    # Filter data for the specified location
    location_df = forecast_df[forecast_df['location'] == location_name]
    
    if location_df.empty:
        print(f"No data found for location '{location_name}'")
        return
    
    # Sort by datetime
    location_df = location_df.sort_values('datetime')
    
    # Get parameter info
    param_name = PARAM_INFO[param]['name']
    param_unit = PARAM_INFO[param]['unit']
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot time series
    plt.plot(location_df['datetime'], location_df[param], marker='o', linestyle='-', markersize=4)
    
    # Add titles and labels
    plt.title(f"{param_name} Forecast for {location_name}")
    plt.xlabel('Date/Time')
    plt.ylabel(f"{param_name} ({param_unit})")
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    filename = f"{location_name.replace(' ', '_')}_{param}_forecast.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()
    
    print(f"Saved time series forecast plot for {param_name} in {location_name}")


def plot_weather_map(forecast_df, param, timestamp=None):
    """
    Plot a weather map for a specific parameter and time.
    
    Args:
        forecast_df (pd.DataFrame): Dataframe with forecast data
        param (str): Weather parameter to plot
        timestamp (datetime, optional): Specific timestamp to plot.
                                       If None, uses the first timestamp.
    """
    # If timestamp not provided, use the first one
    if timestamp is None:
        timestamp = forecast_df['datetime'].min()
    
    # Filter data for the specified timestamp (find the closest one)
    forecast_df['time_diff'] = abs(forecast_df['datetime'] - timestamp)
    closest_time_df = forecast_df.loc[forecast_df.groupby('location')['time_diff'].idxmin()]
    closest_time_df = closest_time_df.drop(columns='time_diff')
    
    # Check if we have enough data points
    if len(closest_time_df) < 3:
        print(f"Not enough data points for map visualization (need at least 3)")
        return
    
    # Get parameter info
    param_name = PARAM_INFO[param]['name']
    param_unit = PARAM_INFO[param]['unit']
    cmap_name = PARAM_INFO[param]['cmap']
    
    # Create figure with cartopy projection
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Get coordinate bounds
    min_lon = closest_time_df['lon'].min() - 5
    max_lon = closest_time_df['lon'].max() + 5
    min_lat = closest_time_df['lat'].min() - 5
    max_lat = closest_time_df['lat'].max() + 5
    
    # Set map extent
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Create scatter plot colored by parameter value
    scatter = ax.scatter(
        closest_time_df['lon'], 
        closest_time_df['lat'],
        c=closest_time_df[param],
        s=100,
        cmap=cmap_name,
        transform=ccrs.PlateCarree(),
        edgecolor='black',
        zorder=10
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, pad=0.1)
    cbar.set_label(f"{param_name} ({param_unit})")
    
    # Add location labels
    for _, row in closest_time_df.iterrows():
        plt.text(
            row['lon'] + 0.3, 
            row['lat'] + 0.3, 
            row['location'],
            transform=ccrs.PlateCarree(),
            fontsize=10,
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
        )
    
    # Add title with timestamp
    time_str = timestamp.strftime('%Y-%m-%d %H:%M')
    plt.title(f"{param_name} Forecast - {time_str}")
    
    # Save the figure
    plt.tight_layout()
    filename = f"weather_map_{param}_{timestamp.strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()
    
    print(f"Saved weather map for {param_name} at {time_str}")


def create_daily_forecast_summary(forecast_df, location_name):
    """
    Create a daily summary visualization of the forecast for a location.
    
    Args:
        forecast_df (pd.DataFrame): Dataframe with forecast data
        location_name (str): Name of location to visualize
    """
    # Filter data for the specified location
    location_df = forecast_df[forecast_df['location'] == location_name]
    
    if location_df.empty:
        print(f"No data found for location '{location_name}'")
        return
    
    # Sort by datetime
    location_df = location_df.sort_values('datetime')
    
    # Create a datetime column with just the date (no time)
    location_df['date'] = location_df['datetime'].dt.date
    
    # Group by date to get daily summaries
    daily_summary = location_df.groupby('date').agg({
        'temp_c': ['min', 'max', 'mean'],
        'humidity': 'mean',
        'wind_kph': 'mean',
        'precip_mm': 'sum',
        'cloud': 'mean'
    })
    
    # Flatten the multi-index columns
    daily_summary.columns = ['_'.join(col).strip() for col in daily_summary.columns.values]
    
    # Reset index to convert date back to a column
    daily_summary = daily_summary.reset_index()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"7-Day Weather Forecast for {location_name}", fontsize=16, y=0.98)
    
    # Plot temperature
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(daily_summary['date'], daily_summary['temp_c_max'], 'r-', marker='o', label='Max')
    ax1.plot(daily_summary['date'], daily_summary['temp_c_mean'], 'g-', marker='o', label='Avg')
    ax1.plot(daily_summary['date'], daily_summary['temp_c_min'], 'b-', marker='o', label='Min')
    ax1.set_ylabel("Temperature (°C)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot precipitation and cloud cover
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.bar(daily_summary['date'], daily_summary['precip_mm_sum'], color='blue', alpha=0.7, label='Precipitation')
    ax2.set_ylabel("Precipitation (mm)")
    ax2.grid(True, alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(daily_summary['date'], daily_summary['cloud_mean'], 'gray', marker='s', label='Cloud Cover')
    ax2_twin.set_ylabel("Cloud Cover (%)")
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot humidity and wind
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(daily_summary['date'], daily_summary['humidity_mean'], 'b-', marker='o', label='Humidity')
    ax3.set_ylabel("Humidity (%)")
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(daily_summary['date'], daily_summary['wind_kph_mean'], 'g-', marker='s', label='Wind Speed')
    ax3_twin.set_ylabel("Wind Speed (kph)")
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Format x-axis dates
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%b %d'))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the figure
    filename = f"{location_name.replace(' ', '_')}_daily_forecast.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()
    
    print(f"Saved daily forecast summary for {location_name}")


def create_hourly_forecast_chart(forecast_df, location_name, day_offset=0):
    """
    Create an hourly forecast chart for a specific day.
    
    Args:
        forecast_df (pd.DataFrame): Dataframe with forecast data
        location_name (str): Name of location to visualize
        day_offset (int): Day offset from today (0=today, 1=tomorrow, etc.)
    """
    # Filter data for the specified location
    location_df = forecast_df[forecast_df['location'] == location_name]
    
    if location_df.empty:
        print(f"No data found for location '{location_name}'")
        return
    
    # Get the target date
    target_date = (datetime.now() + timedelta(days=day_offset)).date()
    
    # Filter data for the target date
    location_df['date'] = location_df['datetime'].dt.date
    day_df = location_df[location_df['date'] == target_date]
    
    if day_df.empty:
        print(f"No data found for {location_name} on {target_date}")
        return
    
    # Sort by hour
    day_df = day_df.sort_values('datetime')
    
    # Extract hour for plotting
    day_df['hour'] = day_df['datetime'].dt.hour
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    date_str = target_date.strftime('%A, %B %d, %Y')
    fig.suptitle(f"Hourly Weather Forecast for {location_name} - {date_str}", fontsize=16, y=0.98)
    
    # Plot temperature
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(day_df['hour'], day_df['temp_c'], 'r-', marker='o', linewidth=2)
    ax1.set_ylabel("Temperature (°C)")
    ax1.grid(True, alpha=0.3)
    
    # Plot humidity
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(day_df['hour'], day_df['humidity'], 'b-', marker='o', linewidth=2)
    ax2.set_ylabel("Humidity (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Plot wind
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(day_df['hour'], day_df['wind_kph'], 'g-', marker='o', linewidth=2)
    ax3.set_ylabel("Wind Speed (kph)")
    ax3.grid(True, alpha=0.3)
    
    # Plot precipitation and cloud cover
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.bar(day_df['hour'], day_df['precip_mm'], color='blue', alpha=0.7, label='Precipitation')
    ax4.set_ylabel("Precipitation (mm)")
    ax4.set_xlabel("Hour of Day")
    ax4.grid(True, alpha=0.3)
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(day_df['hour'], day_df['cloud'], 'gray', marker='s', label='Cloud Cover')
    ax4_twin.set_ylabel("Cloud Cover (%)")
    
    # Combine legends for the last subplot
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Set common x-axis properties
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the figure
    filename = f"{location_name.replace(' ', '_')}_hourly_forecast_{target_date.strftime('%Y%m%d')}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()
    
    print(f"Saved hourly forecast chart for {location_name} on {date_str}")


def main():
    """Main function to create weather visualizations."""
    print("ClimaTrend Neural Forecast - Weather Visualization")
    
    try:
        # Load forecast data
        forecast_df = load_forecast_data()
        
        # Sample locations
        locations = ['New York', 'Los Angeles', 'Chicago', 'Miami', 'Seattle']
        
        # Create visualizations for each location
        for location in locations:
            print(f"\nCreating visualizations for {location}...")
            
            # Create time series forecasts for each parameter
            for param in WEATHER_PARAMS:
                plot_time_series_forecast(forecast_df, location, param)
            
            # Create daily forecast summary
            create_daily_forecast_summary(forecast_df, location)
            
            # Create hourly forecast charts for today and tomorrow
            create_hourly_forecast_chart(forecast_df, location, day_offset=0)
            create_hourly_forecast_chart(forecast_df, location, day_offset=1)
        
        # Create weather maps for different parameters
        print("\nCreating weather maps...")
        
        # Get a few times to create maps for
        times = [
            datetime.now(),  # Now
            datetime.now() + timedelta(days=1),  # Tomorrow
            datetime.now() + timedelta(days=3)   # 3 days from now
        ]
        
        for time in times:
            for param in ['temp_c', 'precip_mm', 'cloud']:
                plot_weather_map(forecast_df, param, time)
        
        print("\nWeather visualization complete.")
        
    except Exception as e:
        print(f"Error during weather visualization: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()