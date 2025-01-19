import rasterio
from rasterio.transform import from_origin
import numpy as np

# Define raster properties
rows, cols = 100, 100  # Grid size
ndvi_data = np.random.uniform(0.2, 0.8, size=(rows, cols))  # Random NDVI values
transform = from_origin(8.5, 7.7, 0.01, 0.01)  # Approximate coordinates for Makurdi

# Save as GeoTIFF
with rasterio.open(
    "makurdi_ndvi_sample.tif", "w", driver="GTiff",
    height=rows, width=cols, count=1, dtype=ndvi_data.dtype.name,
    crs="+proj=latlong", transform=transform
) as dst:
    dst.write(ndvi_data, 1)

print("Sample GeoTIFF data created as makurdi_ndvi_sample.tif")
