"""
Satellite Imagery Download Module

This module handles downloading satellite imagery from Microsoft Planetary Computer,
Google Earth Engine, and Sentinel-2 cloudless tiles for the geospatial foundation models.
"""

import math
import os
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
from io import BytesIO
import numpy as np
import rasterio
from rasterio import windows, features, warp
from shapely.geometry import Point
from shapely.geometry import mapping
import pystac_client
import planetary_computer
from retrying import retry
import logging
import torch
import torch.nn.functional as F
import requests
from PIL import Image

from .paths import get_cache_path

logger = logging.getLogger(__name__)


class ImageryDownloader:
    """Base class for satellite imagery downloading."""
    
    def __init__(self, cache_dir: Optional[Union[str, os.PathLike]] = None):
        self.cache_dir = str(get_cache_path(cache_dir))
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def download_for_locations(self, 
                             latitudes: List[float],
                             longitudes: List[float],
                             datetimes: List[datetime],
                             bands: List[str],
                             image_size: Tuple[int, int] = (64, 64),
                             temporal_window_days: int = 30,
                             max_cloud_cover: float = 0.1) -> Dict[str, Any]:
        """Download imagery for multiple locations."""
        raise NotImplementedError
        

class PlanetaryComputerDownloader(ImageryDownloader):
    """Downloads satellite imagery from Microsoft Planetary Computer."""
    
    # Retry configuration for robustness
    NUM_RETRIES = 5
    WAIT_INTERVAL = 1000  # milliseconds
    
    def __init__(self, cache_dir: Optional[Union[str, os.PathLike]] = None, api_key: Optional[str] = None):
        super().__init__(cache_dir)
        
        # Set up Planetary Computer client
        if api_key:
            planetary_computer.settings.set_subscription_key(api_key)
        
        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )
        
        # Band mapping for different models
        self.band_configs = {
            "satbird": ["B02", "B03", "B04", "B08"],  # Blue, Green, Red, NIR
            "galileo": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],  # Full S2 bands
            "prithvi_v2": ["B02", "B03", "B04", "B08", "B11", "B12"]  # HLS bands
        }
    
    def download_for_locations(self, 
                             latitudes: List[float],
                             longitudes: List[float],
                             datetimes: List[datetime],
                             bands: List[str],
                             image_size: Tuple[int, int] = (64, 64),
                             temporal_window_days: int = 30,
                             max_cloud_cover: float = 0.1) -> Dict[str, Any]:
        """
        Download Sentinel-2 imagery for multiple locations.
        
        Args:
            latitudes: List of latitude coordinates
            longitudes: List of longitude coordinates
            datetimes: List of target datetime objects
            bands: List of Sentinel-2 band names to download
            image_size: Output image size (height, width) in pixels
            temporal_window_days: Days around target date to search
            max_cloud_cover: Maximum cloud cover fraction (0-1)
            
        Returns:
            Dictionary containing imagery data and metadata
        """
        results = {
            "imagery": [],
            "metadata": [],
            "coordinates": list(zip(latitudes, longitudes)),
            "datetimes": datetimes,
            "bands": bands
        }
        
        for i, (lat, lon, target_date) in enumerate(zip(latitudes, longitudes, datetimes)):
            try:
                logger.info(f"Downloading imagery for location {i+1}/{len(latitudes)}: ({lat}, {lon})")
                
                # Create bounding box around the point
                bbox = self._create_bbox(lat, lon, image_size)
                
                # Download imagery for this location
                imagery_data, metadata = self._download_single_location(
                    bbox, target_date, bands, temporal_window_days, max_cloud_cover
                )
                
                # Resize imagery to the requested size
                if imagery_data is not None:
                    imagery_data = self._resize_imagery(imagery_data, image_size)
                
                results["imagery"].append(imagery_data)
                results["metadata"].append(metadata)
                
            except Exception as e:
                logger.error(f"Failed to download imagery for location {i}: {e}")
                # Create placeholder with NaN values
                placeholder = np.full((len(bands), image_size[0], image_size[1]), np.nan)
                results["imagery"].append(placeholder)
                results["metadata"].append({"error": str(e), "success": False})
        
        return results
    
    def _create_bbox(self, lat: float, lon: float, image_size: Tuple[int, int], 
                     pixel_size: float = 10.0) -> Dict[str, Any]:
        """Create a bounding box around a point."""
        # Calculate the extent in meters
        extent_m = max(image_size) * pixel_size / 2

        # Add some buffer
        extent_m = extent_m * 2
        
        # Convert to degrees (approximate)
        lat_deg = extent_m / 111000  # ~111km per degree latitude
        lon_deg = extent_m / (111000 * np.cos(np.radians(lat)))  # longitude depends on latitude
        
        # Create geometry
        point = Point(lon, lat)
        bbox_coords = [
            [lon - lon_deg, lat - lat_deg],
            [lon + lon_deg, lat - lat_deg], 
            [lon + lon_deg, lat + lat_deg],
            [lon - lon_deg, lat + lat_deg],
            [lon - lon_deg, lat - lat_deg]
        ]
        
        geometry = {
            "type": "Polygon",
            "coordinates": [bbox_coords]
        }
        
        return geometry
    
    def _resize_imagery(self, imagery: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize imagery to target size using bilinear interpolation."""
        if imagery is None:
            return imagery
            
        # Check if resizing is needed
        current_shape = imagery.shape
        if len(current_shape) == 3 and current_shape[1:] == target_size:
            return imagery
        
        # Convert to tensor for resizing: [bands, height, width] -> [1, bands, height, width]
        imagery_tensor = torch.from_numpy(imagery).float().unsqueeze(0)

        # Center crop if larger than target size
        if current_shape[1] > target_size[0] or current_shape[2] > target_size[1]:
            top = (current_shape[1] - target_size[0]) // 2
            left = (current_shape[2] - target_size[1]) // 2
            imagery_tensor = imagery_tensor[:, :, top:top+target_size[0], left:left+target_size[1]]
        
        # Resize using bilinear interpolation
        resized_tensor = F.interpolate(
            imagery_tensor, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert back to numpy: [1, bands, height, width] -> [bands, height, width]
        resized_imagery = resized_tensor.squeeze(0).numpy()
        
        return resized_imagery
    
    @retry(stop_max_attempt_number=NUM_RETRIES, wait_fixed=WAIT_INTERVAL)
    def _download_single_location(self, 
                                geometry: Dict[str, Any],
                                target_date: datetime,
                                bands: List[str],
                                temporal_window_days: int,
                                max_cloud_cover: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Download imagery for a single location."""
        
        # Try progressively larger temporal windows
        current_window = temporal_window_days
        max_window = temporal_window_days * 4  # Maximum 4x the original window
        
        items = None
        while current_window <= max_window:
            # Create time range
            start_date = target_date - timedelta(days=current_window)
            end_date = target_date + timedelta(days=current_window)
            time_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            
            # Search for imagery
            search = self.catalog.search(
                collections=["sentinel-2-l2a"],
                intersects=geometry,
                datetime=time_range,
                query={"eo:cloud_cover": {"lt": max_cloud_cover * 100}}
            )
            
            items = search.get_all_items()
            
            if items:
                if current_window > temporal_window_days:
                    logger.info(f"Found imagery with expanded temporal window of {current_window} days")
                break
            else:
                logger.warning(f"No imagery found with {current_window} day window, trying {current_window * 2} days")
                current_window *= 2
                
        if not items:
            raise RuntimeError(f"No Sentinel-2 imagery found within {max_window} days of target date with {max_cloud_cover*100}% cloud cover")
        
        # Sign items for access
        items = planetary_computer.sign(items)
        
        # Select the least cloudy item closest to target date
        def score_item(item):
            from dateutil import parser
            item_date = parser.parse(item.properties["datetime"])
            # Make target_date timezone-aware if it isn't
            if target_date.tzinfo is None:
                target_date_aware = target_date.replace(tzinfo=item_date.tzinfo)
            else:
                target_date_aware = target_date
            time_diff = abs((item_date - target_date_aware).total_seconds())
            cloud_cover = item.properties["eo:cloud_cover"]
            # Combine time difference (in days) and cloud cover
            return time_diff / (24 * 3600) + cloud_cover / 10
            
        best_item = min(items, key=score_item)
        
        # Download bands
        band_data = []
        target_shape = None
        
        for band in bands:
            if band not in best_item.assets:
                raise ValueError(f"Band {band} not available in the selected item")
                
            asset_href = best_item.assets[band].href
            with rasterio.open(asset_href) as ds:
                # Calculate the window for our area of interest
                aoi_bounds = features.bounds(geometry)
                warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
                aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
                
                # Read the data
                data = ds.read(window=aoi_window)
                band_array = data.squeeze()
                
                # Set target shape from first band
                if target_shape is None:
                    target_shape = band_array.shape
                    logger.debug(f"Set target shape from first band {band}: {target_shape}")
                
                # Ensure consistent shape across all bands
                if band_array.shape != target_shape:
                    logger.warning(f"Band {band} shape {band_array.shape} differs from target {target_shape}, resizing")
                    # Resize using PyTorch interpolation if shapes differ
                    band_tensor = torch.from_numpy(band_array).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                    resized_tensor = F.interpolate(
                        band_tensor, 
                        size=target_shape, 
                        mode='bilinear', 
                        align_corners=False
                    )
                    band_array = resized_tensor.squeeze(0).squeeze(0).numpy()
                
                band_data.append(band_array)
        
        # Stack bands - all should have the same shape now
        try:
            imagery = np.stack(band_data, axis=0)
        except ValueError as e:
            # If stacking still fails, provide debug information
            shapes = [arr.shape for arr in band_data]
            logger.error(f"Failed to stack bands with shapes: {shapes}")
            logger.error(f"Target shape was: {target_shape}")
            raise RuntimeError(f"Band stacking failed: {e}. Band shapes: {shapes}") from e
        
        # Metadata
        metadata = {
            "success": True,
            "item_id": best_item.id,
            "datetime": best_item.properties["datetime"],
            "cloud_cover": best_item.properties["eo:cloud_cover"],
            "platform": best_item.properties.get("platform", "unknown"),
            "processing_level": best_item.properties.get("processing:level", "unknown"),
            "shape": imagery.shape
        }
        
        return imagery, metadata
    
    def get_bands_for_model(self, model_name: str) -> List[str]:
        """Get the appropriate bands for a specific model."""
        model_name = model_name.lower()
        if model_name in self.band_configs:
            return self.band_configs[model_name]
        else:
            raise ValueError(f"Unknown model: {model_name}")


class EarthEngineDownloader(ImageryDownloader):
    """Downloads satellite imagery from Google Earth Engine."""
    
    def __init__(self, cache_dir: Optional[Union[str, os.PathLike]] = None):
        super().__init__(cache_dir)
        # Initialize Earth Engine
        try:
            import ee
            ee.Initialize()
            self.ee = ee
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Earth Engine: {e}")
        
        # Band mapping for Earth Engine Sentinel-2 collection  
        self.ee_band_mapping = {
            "B02": "B2",  # Blue
            "B03": "B3",  # Green  
            "B04": "B4",  # Red
            "B05": "B5",  # Red Edge 1
            "B06": "B6",  # Red Edge 2
            "B07": "B7",  # Red Edge 3
            "B08": "B8",  # NIR
            "B8A": "B8A", # Red Edge 4
            "B11": "B11", # SWIR 1
            "B12": "B12"  # SWIR 2
        }
    
    def download_for_locations(self, 
                             latitudes: List[float],
                             longitudes: List[float], 
                             datetimes: List[datetime],
                             bands: List[str],
                             image_size: Tuple[int, int] = (64, 64),
                             temporal_window_days: int = 30,
                             max_cloud_cover: float = 0.1) -> Dict[str, Any]:
        """
        Download Sentinel-2 imagery using Google Earth Engine.
        
        Args:
            latitudes: List of latitude coordinates
            longitudes: List of longitude coordinates
            datetimes: List of target datetime objects
            bands: List of Sentinel-2 band names to download
            image_size: Output image size (height, width) in pixels
            temporal_window_days: Days around target date to search
            max_cloud_cover: Maximum cloud cover fraction (0-1)
            
        Returns:
            Dictionary containing imagery data and metadata
        """
        results = {
            "imagery": [],
            "metadata": [],
            "coordinates": list(zip(latitudes, longitudes)),
            "datetimes": datetimes,
            "bands": bands
        }
        
        # Convert bands to Earth Engine naming convention
        ee_bands = [self.ee_band_mapping.get(band, band) for band in bands]
        
        for i, (lat, lon, target_date) in enumerate(zip(latitudes, longitudes, datetimes)):
            try:
                logger.info(f"Downloading EE imagery for location {i+1}/{len(latitudes)}: ({lat}, {lon})")
                
                # Create point geometry
                point = self.ee.Geometry.Point(lon, lat)
                
                # Create region around point
                region = point.buffer(image_size[0] * 5)  # 5m per pixel buffer roughly
                
                # Download imagery for this location
                imagery_data, metadata = self._download_single_location_ee(
                    point, region, target_date, ee_bands, temporal_window_days, max_cloud_cover
                )
                
                # Resize imagery to the requested size
                if imagery_data is not None:
                    imagery_data = self._resize_imagery(imagery_data, image_size)
                
                results["imagery"].append(imagery_data)
                results["metadata"].append(metadata)
                
            except Exception as e:
                logger.error(f"Failed to download EE imagery for location {i}: {e}")
                # Create placeholder with NaN values
                placeholder = np.full((len(bands), image_size[0], image_size[1]), np.nan)
                results["imagery"].append(placeholder)
                results["metadata"].append({"error": str(e), "success": False})
        
        return results


class SentinelCloudlessDownloader(ImageryDownloader):
    """Downloads RGB imagery from the Sentinel-2 cloudless tiles served by maps.eox.at."""

    def __init__(
        self,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        layer_template: str = "s2cloudless-{year}_3857",
        pixel_size_meters: float = 10.0,
        request_timeout: int = 30,
        session: Optional[requests.Session] = None,
        min_year: int = 2016,
        max_year: int = 2024,
        default_year: int = 2024,
    ):
        super().__init__(cache_dir)
        self.layer_template = layer_template
        self.pixel_size_meters = pixel_size_meters
        self.request_timeout = request_timeout
        self.base_url = "https://tiles.maps.eox.at/wms"
        self.session = session or requests.Session()
        self.expected_bands = ["R", "G", "B"]
        self.min_year = int(min_year)
        self.max_year = int(max_year)
        self.default_year = int(np.clip(default_year, self.min_year, self.max_year))

    def download_for_locations(
        self,
        latitudes: List[float],
        longitudes: List[float],
        datetimes: List[datetime],
        bands: List[str],
        image_size: Tuple[int, int] = (224, 224),
        temporal_window_days: int = 30,
        max_cloud_cover: float = 0.1,
    ) -> Dict[str, Any]:
        """Download RGB patches centered on each location."""
        height, width = image_size
        height = int(height)
        width = int(width)

        results = {
            "imagery": [],
            "metadata": [],
            "coordinates": list(zip(latitudes, longitudes)),
            "datetimes": datetimes,
            "bands": self.expected_bands,
        }

        for idx, (lat, lon, target_date) in enumerate(zip(latitudes, longitudes, datetimes)):
            year = self._infer_year(target_date)
            layer = self.layer_template.format(year=year)
            try:
                imagery = self._fetch_patch(lat, lon, height, width, layer)
                metadata = {
                    "success": True,
                    "datetime": target_date.isoformat(),
                    "layer": layer,
                    "pixel_size_meters": self.pixel_size_meters,
                    "shape": imagery.shape,
                    "year": year,
                }
                results["imagery"].append(imagery)
                results["metadata"].append(metadata)
            except Exception as exc:
                logger.error(
                    "Failed to download cloudless imagery for (%s, %s): %s", lat, lon, exc
                )
                placeholder = np.full((3, height, width), np.nan, dtype=np.float32)
                results["imagery"].append(placeholder)
                results["metadata"].append(
                    {
                        "success": False,
                        "error": str(exc),
                        "layer": layer,
                        "year": year,
                    }
                )

        return results

    def _fetch_patch(self, lat: float, lon: float, height: int, width: int, layer: str) -> np.ndarray:
        minx, miny, maxx, maxy = self._compute_bbox(lat, lon, height, width)
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.1.1",
            "LAYERS": layer,
            "STYLES": "",
            "FORMAT": "image/jpeg",
            "SRS": "EPSG:3857",
            "WIDTH": width,
            "HEIGHT": height,
            "BBOX": f"{minx},{miny},{maxx},{maxy}",
        }
        response = self.session.get(
            self.base_url, params=params, timeout=self.request_timeout
        )
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as img:
            rgb = img.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32) / 255.0
            array = np.transpose(array, (2, 0, 1))
        return array

    def _compute_bbox(self, lat: float, lon: float, height: int, width: int) -> Tuple[float, float, float, float]:
        center_x, center_y = self._latlon_to_web_mercator(lat, lon)
        half_width = (width * self.pixel_size_meters) / 2.0
        half_height = (height * self.pixel_size_meters) / 2.0
        return (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        )

    def _latlon_to_web_mercator(self, lat: float, lon: float) -> Tuple[float, float]:
        origin_shift = 20037508.342789244
        x = lon * origin_shift / 180.0
        lat = max(min(lat, 89.9), -89.9)
        y = math.log(math.tan((90 + lat) * math.pi / 360.0)) * origin_shift / math.pi
        return x, y

    def _infer_year(self, target_date: datetime) -> int:
        if isinstance(target_date, datetime):
            year = target_date.year
        else:
            year = self.default_year
        return int(np.clip(year, self.min_year, self.max_year))
    
    def _download_single_location_ee(self,
                                   point: Any,
                                   region: Any, 
                                   target_date: datetime,
                                   bands: List[str],
                                   temporal_window_days: int,
                                   max_cloud_cover: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Download imagery for a single location using Earth Engine."""
        
        # Try progressively larger temporal windows
        current_window = temporal_window_days
        max_window = temporal_window_days * 4  # Maximum 4x the original window
        
        collection = None
        while current_window <= max_window:
            # Create time range
            start_date = target_date - timedelta(days=current_window)
            end_date = target_date + timedelta(days=current_window)
            
            # Get Sentinel-2 collection
            collection = (self.ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                         .filterBounds(point)
                         .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                         .filter(self.ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover * 100)))
            
            # Check if collection is empty
            collection_size = collection.size()
            if collection_size.getInfo() > 0:
                if current_window > temporal_window_days:
                    logger.info(f"Found imagery with expanded temporal window of {current_window} days")
                break
            else:
                logger.warning(f"No imagery found with {current_window} day window, trying {current_window * 2} days")
                current_window *= 2
                
        if collection is None or collection.size().getInfo() == 0:
            raise RuntimeError(f"No Sentinel-2 imagery found within {max_window} days of target date with {max_cloud_cover*100}% cloud cover")
        
        # Select the image closest to target date with least cloud cover
        def add_date_band(image):
            target_millis = self.ee.Date(target_date.strftime('%Y-%m-%d')).millis()
            image_millis = image.date().millis()
            time_diff = image_millis.subtract(target_millis).abs()
            return image.set('time_diff', time_diff)
        
        collection_with_dates = collection.map(add_date_band)
        
        # Sort by cloud cover, then by time difference
        best_image = (collection_with_dates
                     .sort('CLOUDY_PIXEL_PERCENTAGE')
                     .sort('time_diff')
                     .first())
        
        # Select bands and scale to 0-1
        image = best_image.select(bands).divide(10000)
        
        # Define export parameters
        export_params = {
            'image': image,
            'region': region,
            'scale': 10,  # 10m resolution
            'maxPixels': 1e9,
            'format': 'GeoTIFF'
        }
        
        # Export to numpy array (this is a simplified approach)
        # In practice, you might want to export to Google Drive/Cloud Storage
        # and download from there for larger areas
        try:
            # Get image as numpy array for small regions
            image_data = image.sampleRectangle(region=region, defaultValue=0)
            
            # Convert to numpy arrays
            band_arrays = []
            for band in bands:
                band_data = np.array(image_data.select([band]).getInfo()['properties'][band])
                band_arrays.append(band_data)
            
            imagery = np.stack(band_arrays, axis=0)
            
            # Get metadata
            image_info = best_image.getInfo()
            metadata = {
                "success": True,
                "image_id": image_info['id'],
                "date": image_info['properties'].get('PRODUCT_ID', 'unknown'),
                "cloud_cover": image_info['properties'].get('CLOUDY_PIXEL_PERCENTAGE', 'unknown'),
                "platform": image_info['properties'].get('SPACECRAFT_NAME', 'unknown'),
                "shape": imagery.shape
            }
            
            return imagery, metadata
            
        except Exception as e:
            # Fallback: try median composite for the time period
            logger.warning(f"Single image download failed, trying median composite: {e}")
            
            # Create median composite
            composite = collection.median().select(bands).divide(10000)
            
            # Sample the composite
            composite_data = composite.sampleRectangle(region=region, defaultValue=0)
            
            # Convert to numpy arrays
            band_arrays = []
            for band in bands:
                band_data = np.array(composite_data.select([band]).getInfo()['properties'][band])
                band_arrays.append(band_data)
            
            imagery = np.stack(band_arrays, axis=0)
            
            metadata = {
                "success": True,
                "image_id": "median_composite",
                "date": f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}",
                "cloud_cover": "composite",
                "platform": "Sentinel-2",
                "shape": imagery.shape,
                "method": "median_composite"
            }
            
            return imagery, metadata
    
    def _resize_imagery(self, imagery: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize imagery to target size using bilinear interpolation."""
        if imagery is None:
            return imagery
            
        # Check if resizing is needed
        current_shape = imagery.shape
        if len(current_shape) == 3 and current_shape[1:] == target_size:
            return imagery
        
        # Convert to tensor for resizing: [bands, height, width] -> [1, bands, height, width]
        imagery_tensor = torch.from_numpy(imagery).float().unsqueeze(0)
        
        # Resize using bilinear interpolation
        resized_tensor = F.interpolate(
            imagery_tensor, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert back to numpy: [1, bands, height, width] -> [bands, height, width]
        resized_imagery = resized_tensor.squeeze(0).numpy()
        
        return resized_imagery


def create_downloader(source: str, **kwargs) -> ImageryDownloader:
    """Factory function to create appropriate downloader."""
    if source == "planetary_computer":
        return PlanetaryComputerDownloader(**kwargs)
    elif source == "earth_engine":
        return EarthEngineDownloader(**kwargs)
    elif source == "sentinel_cloudless":
        return SentinelCloudlessDownloader(**kwargs)
    else:
        raise ValueError(f"Unknown imagery source: {source}")


if __name__ == "__main__":
    # Test the downloader
    from datetime import datetime
    
    # Test coordinates
    lats = [40.7128]  # New York
    lons = [-74.0060]
    times = [datetime(2023, 6, 15)]
    
    # Create downloader
    downloader = PlanetaryComputerDownloader()
    
    # Get bands for SatBird
    bands = downloader.get_bands_for_model("satbird")
    print(f"SatBird bands: {bands}")
    
    # Test download (will fail without proper API key)
    try:
        result = downloader.download_for_locations(lats, lons, times, bands)
        print(f"Downloaded imagery shape: {result['imagery'][0].shape}")
    except Exception as e:
        print(f"Download test failed (expected without API key): {e}")
