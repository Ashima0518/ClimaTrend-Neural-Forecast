# Installation Guide for ClimaTrend Neural Forecast

This document provides detailed instructions for setting up the ClimaTrend Neural Forecast system on your machine.

## Prerequisites

- Python 3.8 or later
- pip (Python package installer)
- Optional: NVIDIA GPU with CUDA support for faster model training

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/climatrend-neural-forecast.git
cd climatrend-neural-forecast
```

### 2. Create a Virtual Environment (Recommended)

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### For GPU Support (Optional):
If you have an NVIDIA GPU and want to use it for training, install the GPU version of TensorFlow:

```bash
pip uninstall tensorflow
pip install tensorflow-gpu==2.13.0
```

### 4. Set Up Environment Variables (Optional)

```bash
cp example.env .env
```

Then edit the `.env` file to configure model parameters if needed. The default values will work fine for most users.

### 5. Prepare Data Directories

The system will automatically create necessary directories, but you can also set them up manually:

```bash
mkdir -p data/{raw,processed}/{weather_data,satellite_images}
mkdir -p models
mkdir -p logs
mkdir -p plots
```

## Installing Additional Requirements

### Cartopy (for Weather Maps)

Cartopy may require additional system libraries:

#### On Ubuntu/Debian:
```bash
sudo apt-get install libgeos-dev libproj-dev
```

#### On macOS (with Homebrew):
```bash
brew install geos proj
```

#### On Windows:
For Windows, it's recommended to install Cartopy via conda:
```bash
conda install -c conda-forge cartopy
```

## Testing Your Installation

After installation, you can run a simple test to verify everything is set up correctly:

```bash
python main.py --visualize
```

This will generate sample weather visualizations using synthetic data.

To run the complete pipeline:

```bash
python main.py --all
```

## Understanding the Synthetic Data

ClimaTrend Neural Forecast uses realistic synthetic data:

1. **Synthetic Weather Data**: The system generates weather data with realistic patterns including:
   - Seasonal variations based on time of year
   - Temperature patterns based on latitude
   - Correlated weather parameters (humidity, precipitation, cloud cover)

2. **Synthetic Satellite Images**: The system generates satellite-like images with:
   - Geographic features based on location
   - Seasonal variations in vegetation and cloud coverage
   - Realistic cloud and terrain patterns

All synthetic data is deterministic using timestamp-based seeds, so results are reproducible.

## Troubleshooting

- **ImportError with TensorFlow**: Make sure you have the compatible version of CUDA and cuDNN if using GPU support.
- **Cartopy Installation Issues**: If you encounter problems installing Cartopy, consider using conda instead of pip for this package.

## Additional Resources

- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Cartopy Documentation: [https://scitools.org.uk/cartopy/docs/latest/](https://scitools.org.uk/cartopy/docs/latest/) 