# Advanced GIS Application Suite

Welcome to the **Advanced GIS Application Suite**, a collection of tools tailored for agricultural and environmental applications. This README file provides an overview of the app's features, installation instructions, and usage guidelines.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Modules Overview](#modules-overview)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Mineral Detection:** Analyze mineral data from hyperspectral images or CSV files.
- **Season Detection:** Identify the start and end dates of seasons based on temporal raster datasets.
- **Live Watch Locations:** Monitor GPS locations and visualize data on interactive maps.
- **Harvesting Time Determination:** Detect optimal harvesting times using NDVI (Normalized Difference Vegetation Index).
- **Flood Detection:** Identify flood-prone areas using DEM (Digital Elevation Model) data.
- **Crop Detection:** Detect and visualize crop health through images, real-time camera feeds, and NDVI analysis.

---

## Requirements
The application requires the following:

- **Python:** Version 3.8 or above
- **Libraries:**
  - PyQt5
  - OpenCV
  - Rasterio
  - Matplotlib
  - Numpy
  - Pandas
  - Geopy
  - Folium
  - Scikit-learn

To install all required libraries, use:
```bash
pip install -r requirements.txt
```

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kayung-developer/Advanced-GIS.git
   cd Advanced-GIS
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

---

## Usage
1. Launch the application using the command:
   ```bash
   python main.py
   ```
2. Select the desired module from the application's tabbed interface.
3. Follow the on-screen instructions to load data and perform analyses.

---

## Modules Overview
### 1. Mineral Detection
- **Input:** Hyperspectral GeoTIFF files or CSV files.
- **Output:** Mineral detection results and visualizations.
- **Key Features:**
  - PCA-based mineral analysis.
  - Export results to CSV.
  - Visualize hyperspectral data.

### 2. Season Detection
- **Input:** Temporal raster datasets.
- **Output:** Start and end dates of seasons.
- **Key Features:**
  - Adjustable threshold for season detection.
  - Temporal data analysis with visualizations.

### 3. Live Watch Locations
- **Input:** GPS data in CSV format.
- **Output:** Interactive maps displaying locations.
- **Key Features:**
  - Camera integration for real-time detection.
  - Distance calculation and feature visualization on maps.

### 4. Harvesting Time Determination
- **Input:** NDVI datasets in GeoTIFF or CSV format.
- **Output:** Optimal harvesting time.
- **Key Features:**
  - Adjustable NDVI thresholds.
  - NDVI data visualization.

### 5. Flood Detection
- **Input:** DEM data in GeoTIFF format.
- **Output:** Identified flood-prone areas.
- **Key Features:**
  - Elevation threshold adjustment.
  - Export results to CSV.
  - DEM visualization.

### 6. Crop Detection
- **Input:** Images, real-time camera feed, or NDVI files.
- **Output:** Detected crop health and bounding boxes.
- **Key Features:**
  - Real-time camera-based detection.
  - NDVI-based crop health visualization.
  - Export results to CSV.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

Thank you for using the **Advanced GIS Application Suite**!
