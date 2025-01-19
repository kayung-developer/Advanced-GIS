import numpy as np
import rasterio
from rasterio.transform import from_origin

# Define DEM properties
width = 100  # Number of columns (longitude)
height = 100  # Number of rows (latitude)
pixel_size = 0.01  # Size of each pixel in degrees
top_left_x = 8.5  # Longitude of top-left corner (approx for Makurdi)
top_left_y = 7.5  # Latitude of top-left corner (approx for Makurdi)

# Create synthetic elevation data
# Simulate lower elevation in the center to mimic a flood-prone river area
elevation_data = np.random.uniform(5, 50, size=(height, width))  # Base elevation
river_path = np.exp(-((np.arange(width) - width // 2) ** 2) / 50)
for i in range(height):
    elevation_data[i, :] -= river_path * 30  # Simulate river depression

# Define GeoTIFF metadata
transform = from_origin(top_left_x, top_left_y, pixel_size, pixel_size)
metadata = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "float32",
    "crs": "EPSG:4326",  # WGS84
    "transform": transform,
}

# Save the DEM as a GeoTIFF file
output_file = "makurdi_dem.tif"
with rasterio.open(output_file, "w", **metadata) as dst:
    dst.write(elevation_data.astype("float32"), 1)

print(f"Synthetic DEM file saved as {output_file}")
