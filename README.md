# ClimaTrend Neural Forecast

An advanced weather forecasting system that leverages deep learning with Convolutional Neural Networks (CNNs) to process historical weather data and satellite imagery for real-time and future weather predictions.

## Features

- Generation and processing of 10,000+ synthetic historical weather data points
- Creation and analysis of synthetic satellite images with realistic features
- Real-time weather prediction with hourly updates
- Forecasts up to seven days in advance
- High-resolution image analysis of meteorological patterns
- Scalable model architecture with multiple neural network options

## Key Capabilities

- **Synthetic Data Generation**: Creates realistic weather patterns and satellite imagery based on geographic location and seasonal variations
- **Advanced CNN Models**: Utilizes ResNet and custom CNN architectures to process satellite imagery
- **Hybrid CNN-LSTM Models**: Combines satellite image analysis with time-series predictions
- **Comprehensive Visualizations**: Includes time-series forecasts, weather maps, and daily/hourly summaries

## Technologies

- Python 3.8+
- TensorFlow 2.x
- Keras
- Scikit-learn
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Cartopy

## Project Structure

- `data/`: Directory for datasets (synthetic weather data, synthetic satellite images)
- `models/`: Saved trained models
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `src/`: Source code
  - `data_processing/`: Scripts for data generation and preprocessing
  - `models/`: Model architecture definitions
  - `training/`: Training scripts
  - `evaluation/`: Model evaluation scripts
  - `visualization/`: Data and results visualization
- `plots/`: Generated forecast visualizations

## Setup and Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

```bash
# Clone the repository
git clone https://github.com/yourusername/climatrend-neural-forecast.git
cd climatrend-neural-forecast

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Generation

```bash
python main.py --fetch-data --process-images
```

### Model Training

```bash
python main.py --train
```

### Running Predictions and Visualization

```bash
python main.py --visualize
```

### Complete Pipeline

```bash
python main.py --all
```

## Model Architecture

ClimaTrend Neural Forecast implements several deep learning architectures:

1. **Simple CNN**: A basic convolutional network for satellite image processing
2. **ResNet-based Model**: A transfer learning approach using ResNet50V2
3. **LSTM Model**: For time-series weather data processing
4. **Hybrid CNN-LSTM Model**: Combining image and time-series processing

## Sample Visualizations

The system generates various types of visualizations:

- **Time Series Forecasts**: Detailed predictions of temperature, humidity, and other parameters over time
- **Weather Maps**: Geospatial visualizations using Cartopy showing meteorological conditions
- **Daily Summaries**: Day-by-day weather forecasts with key statistics
- **Hourly Charts**: Detailed hourly breakdowns of weather conditions

## License

This project is licensed under the MIT License - see the LICENSE file for details. 