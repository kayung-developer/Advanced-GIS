import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

# Parameters
width, height = 100, 100  # Define raster size
time_steps = 12  # 12 months
rainfall_min, rainfall_max = 0, 300  # Rainfall range (mm)
dry_season_start, dry_season_end = 5, 10  # June to October are dry months
rain_season_start, rain_season_end = 11, 4  # November to April are rainy months

# Generate synthetic rainfall data
rainfall_data = np.zeros((time_steps, height, width))  # Initialize an empty array

# Simulate rainy and dry seasons
for month in range(time_steps):
    if month >= rain_season_start or month <= rain_season_end:
        rainfall_data[month] = np.random.uniform(rainfall_min, rainfall_max, size=(height, width))  # Rainy season
    else:
        rainfall_data[month] = np.random.uniform(0, 50, size=(height, width))  # Dry season

# Define GeoTIFF metadata
transform = from_origin(west=-8.4, north=7.7, xsize=0.01, ysize=0.01)  # Approximate coordinates for Makurdi, Benue
metadata = {
    'driver': 'GTiff',
    'count': time_steps,
    'width': width,
    'height': height,
    'crs': 'EPSG:4326',  # WGS84
    'dtype': 'float32',
    'nodata': None,
    'transform': transform
}

# Save the synthetic raster data to a GeoTIFF file
output_raster_path = "makurdi_rainfall_data.tif"
with rasterio.open(output_raster_path, 'w', **metadata) as dst:
    for i in range(time_steps):
        dst.write(rainfall_data[i], i + 1)

# Visualize the generated rainfall data for one time step
plt.imshow(rainfall_data[0], cmap='Blues')  # Visualize for the first month (e.g., January)
plt.colorbar(label="Rainfall (mm)")
plt.title("Rainfall Data for Makurdi (January)")
plt.show()

print(f"Sample raster data saved to {output_raster_path}")
