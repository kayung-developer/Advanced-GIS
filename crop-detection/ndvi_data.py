import numpy as np
import rasterio
from rasterio.transform import from_origin

# Create synthetic NDVI data
width, height = 100, 100
ndvi_data = np.random.uniform(-1, 1, (height, width)).astype('float32')

# Save as a GeoTIFF
transform = from_origin(9.5240, 7.7305, 0.0001, 0.0001)  # Makurdi coordinates
with rasterio.open(
    "synthetic_ndvi.tif", "w",
    driver="GTiff", height=height, width=width,
    count=1, dtype="float32", crs="EPSG:4326", transform=transform
) as dst:
    dst.write(ndvi_data, 1)
