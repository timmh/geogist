"""
Environmental Data Downloader

Downloads environmental data (bioclimatic and pedological) for SatBird model
from WorldClim and SoilGrids.
"""

import requests
import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import reproject, Resampling
import rasterio.transform
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import zipfile
from tqdm import tqdm
import tempfile

from geogist.paths import get_cache_path
try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available. Soil data interpolation will be limited.")
    SCIPY_AVAILABLE = False

try:
    import soilgrids
    SOILGRIDS_AVAILABLE = True
except ImportError:
    print("Warning: soilgrids package not available. Soil data extraction will be limited.")
    SOILGRIDS_AVAILABLE = False

class EnvironmentalDataDownloader:
    """Downloads and processes environmental data for SatBird."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache_dir = get_cache_path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # WorldClim bioclimatic variables (19 variables)
        self.bioclim_vars = [
            'bio_1',   # Annual Mean Temperature
            'bio_2',   # Mean Diurnal Range
            'bio_3',   # Isothermality
            'bio_4',   # Temperature Seasonality
            'bio_5',   # Max Temperature of Warmest Month
            'bio_6',   # Min Temperature of Coldest Month
            'bio_7',   # Temperature Annual Range
            'bio_8',   # Mean Temperature of Wettest Quarter
            'bio_9',   # Mean Temperature of Driest Quarter
            'bio_10',  # Mean Temperature of Warmest Quarter
            'bio_11',  # Mean Temperature of Coldest Quarter
            'bio_12',  # Annual Precipitation
            'bio_13',  # Precipitation of Wettest Month
            'bio_14',  # Precipitation of Driest Month
            'bio_15',  # Precipitation Seasonality
            'bio_16',  # Precipitation of Wettest Quarter
            'bio_17',  # Precipitation of Driest Quarter
            'bio_18',  # Precipitation of Warmest Quarter
            'bio_19'   # Precipitation of Coldest Quarter
        ]
        
        # SoilGrids pedological variables (8 variables)
        self.soil_vars = [
            'bdod',    # Bulk density
            'cec',     # Cation exchange capacity
            'cfvo',    # Coarse fragments
            'clay',    # Clay content
            'nitrogen', # Nitrogen
            'ocd',     # Organic carbon density
            'phh2o',   # pH in H2O
            'sand'     # Sand content
        ]
        
    def download_bioclim_data(self) -> bool:
        """Download WorldClim bioclimatic data."""
        print("Downloading WorldClim bioclimatic data...")
        
        bioclim_dir = self.cache_dir / "bioclim"
        bioclim_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        if (bioclim_dir / "wc2.1_30s_bio.zip").exists():
            print("✓ Bioclim data already downloaded")
            return True
            
        # Download WorldClim data (30 arc-seconds resolution)
        url = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_bio.zip"
        zip_path = bioclim_dir / "wc2.1_30s_bio.zip"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading bioclim",
                total=total_size,
                unit='B',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(bioclim_dir)
            
            print("✓ Bioclim data downloaded and extracted")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download bioclim data: {e}")
            return False
    
    def download_soil_data(self) -> bool:
        """Download SoilGrids pedological data via REST API."""
        print("Note: SoilGrids requires REST API calls for specific locations")
        print("Using environmental data extraction will create location-specific soil data")
        
        soil_dir = self.cache_dir / "soil"
        soil_dir.mkdir(exist_ok=True)
        
        print("✓ Soil data directory prepared")
        return True
    
    
    def extract_environmental_data(self, 
                                 latitudes: List[float], 
                                 longitudes: List[float],
                                 buffer_size: int = 64) -> List[Dict[str, Any]]:
        """
        Extract environmental data for given coordinates.
        
        Args:
            latitudes: List of latitude coordinates
            longitudes: List of longitude coordinates  
            buffer_size: Size of extraction window (pixels)
            
        Returns:
            List of environmental data dictionaries
        """
        # Ensure data is downloaded
        self.download_bioclim_data()
        self.download_soil_data()
        
        results = []
        
        for lat, lon in zip(latitudes, longitudes):
            env_data = {
                'bioclim': self._extract_bioclim(lat, lon, buffer_size),
                'soil': self._extract_soil(lat, lon, buffer_size),
                'coordinates': (lat, lon)
            }
            results.append(env_data)
        
        return results
    
    def _extract_bioclim(self, lat: float, lon: float, buffer_size: int) -> np.ndarray:
        """Extract bioclimatic data for a location using rasterio."""
        bioclim_dir = self.cache_dir / "bioclim"
        
        # Check if extracted tif files exist, if not extract from zip
        tif_files = list(bioclim_dir.glob("wc2.1_30s_bio_*.tif"))
        if len(tif_files) == 0:
            # Extract from zip if tif files don't exist
            zip_path = bioclim_dir / "wc2.1_30s_bio.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(bioclim_dir)
        
        # Create output array
        bioclim_data = np.zeros((len(self.bioclim_vars), buffer_size, buffer_size), dtype=np.float32)
        
        # Calculate pixel size for sampling window
        pixel_size = 30 / 3600  # 30 arc-seconds in degrees
        half_window = (buffer_size * pixel_size) / 2
        
        # Define bounding box for extraction
        west = lon - half_window
        east = lon + half_window
        south = lat - half_window
        north = lat + half_window
        
        # Extract data from each bioclim variable
        for i, var in enumerate(self.bioclim_vars):
            var_num = i + 1  # WorldClim uses 1-based indexing
            tif_path = bioclim_dir / f"wc2.1_30s_bio_{var_num}.tif"
            
            try:
                with rasterio.open(tif_path) as src:
                    # Create window for extraction
                    window = rasterio.windows.from_bounds(
                        west, south, east, north, src.transform
                    )
                    
                    # Read data within window
                    data = src.read(1, window=window)
                    
                    # Handle nodata values
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    # Resample to desired buffer size if needed
                    if data.shape != (buffer_size, buffer_size):
                        from rasterio.warp import reproject, Resampling
                        
                        # Create target transform
                        target_transform = rasterio.transform.from_bounds(
                            west, south, east, north, buffer_size, buffer_size
                        )
                        
                        # Reproject to target grid
                        resampled = np.zeros((buffer_size, buffer_size), dtype=np.float32)
                        reproject(
                            data.astype(np.float32),
                            resampled,
                            src_transform=rasterio.windows.transform(window, src.transform),
                            src_crs=src.crs,
                            dst_transform=target_transform,
                            dst_crs=src.crs,
                            resampling=Resampling.bilinear
                        )
                        bioclim_data[i] = resampled
                    else:
                        bioclim_data[i] = data.astype(np.float32)
                        
            except FileNotFoundError:
                print(f"Warning: {tif_path} not found")
                bioclim_data[i] = np.full((buffer_size, buffer_size), np.nan, dtype=np.float32)
            except Exception as e:
                print(f"Warning: Could not read {tif_path}: {e}")
                bioclim_data[i] = np.full((buffer_size, buffer_size), np.nan, dtype=np.float32)
        
        return bioclim_data
    
    def _extract_soil(self, lat: float, lon: float, buffer_size: int) -> np.ndarray:
        """Extract soil data for a location using soilgrids package."""
        soil_dir = self.cache_dir / "soil"
        
        # Create output array
        soil_data = np.zeros((len(self.soil_vars), buffer_size, buffer_size), dtype=np.float32)
        
        # Cache file for this location
        cache_file = soil_dir / f"soil_{lat:.4f}_{lon:.4f}_{buffer_size}.npy"
        
        # Check if cached data exists
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except:
                pass  # Continue to fetch new data if cache is corrupted
        
        if not SOILGRIDS_AVAILABLE:
            print("Warning: soilgrids package not available, returning NaN data")
            soil_data.fill(np.nan)
            return soil_data
        
        print(f"Fetching soil data for lat={lat:.4f}, lon={lon:.4f} with buffer size {buffer_size}")
        
        # Calculate pixel size for sampling window (250m SoilGrids resolution)
        pixel_size = 250 / 111320  # 250m resolution in degrees
        half_window = (buffer_size * pixel_size) / 2
        
        # Define bounding box coordinates
        west = lon - half_window
        south = lat - half_window  
        east = lon + half_window
        north = lat + half_window
        
        # SoilGrids property mapping
        soilgrids_properties = {
            'bdod': 'bdod',       # Bulk density
            'cec': 'cec',         # Cation exchange capacity  
            'cfvo': 'cfvo',       # Coarse fragments
            'clay': 'clay',       # Clay content
            'nitrogen': 'nitrogen', # Nitrogen
            'ocd': 'ocd',         # Organic carbon density
            'phh2o': 'phh2o',     # pH in H2O
            'sand': 'sand'        # Sand content
        }
        
        # Initialize SoilGrids client
        sg = soilgrids.SoilGrids()
        
        # Extract data for each soil variable
        for i, var in enumerate(self.soil_vars):
            try:
                if var in soilgrids_properties:
                    # Use soilgrids package to fetch data
                    sg_property = soilgrids_properties[var]
                    
                    # Use a temporary file for the download
                    with tempfile.NamedTemporaryFile(suffix=".tif", dir=soil_dir, delete=True) as temp_f:
                        temp_file_path = temp_f.name
                        
                        # Get data for 15-30cm depth
                        sg.get_coverage_data(
                            service_id=sg_property,
                            coverage_id=f'{sg_property}_15-30cm_mean',
                            crs='urn:ogc:def:crs:EPSG::4326',
                            west=west,
                            south=south,
                            east=east,
                            north=north,
                            output=temp_file_path,
                            width=buffer_size,
                            height=buffer_size
                        )
                        
                        # Read the temporary file
                        try:
                            with rasterio.open(temp_file_path) as src:
                                # Read the raster data
                                var_data = src.read(1).astype(np.float32)
                                
                                # Handle nodata values
                                if src.nodata is not None:
                                    var_data = np.where(var_data == src.nodata, np.nan, var_data)
                                
                                # Ensure correct shape
                                if var_data.shape != (buffer_size, buffer_size):
                                    # Resize if necessary
                                    if SCIPY_AVAILABLE:
                                        from scipy.ndimage import zoom
                                        zoom_factors = (buffer_size / var_data.shape[0], buffer_size / var_data.shape[1])
                                        var_data = zoom(var_data, zoom_factors, order=1)
                                    else:
                                        # Simple fallback: use mean value
                                        mean_val = np.nanmean(var_data) if not np.all(np.isnan(var_data)) else np.nan
                                        var_data = np.full((buffer_size, buffer_size), mean_val, dtype=np.float32)
                                
                                soil_data[i] = var_data
                            
                        except Exception as read_error:
                            print(f"Warning: Could not read downloaded file for {var}: {read_error}")
                            soil_data[i] = np.full((buffer_size, buffer_size), np.nan, dtype=np.float32)
                else:
                    print(f"Warning: Unknown soil variable {var}")
                    soil_data[i] = np.full((buffer_size, buffer_size), np.nan, dtype=np.float32)
                    
            except Exception as e:
                print(f"Warning: Could not fetch soil data for {var}: {e}")
                # Fill with NaN if fetch fails
                soil_data[i] = np.full((buffer_size, buffer_size), np.nan, dtype=np.float32)
        
        # Cache the result
        try:
            np.save(cache_file, soil_data)
        except:
            pass  # Continue even if caching fails
        
        return soil_data
    
    
    def get_environmental_stats(self) -> Dict[str, Any]:
        """Get normalization statistics for environmental data."""
        # WorldClim BioClim variables global statistics (approximate)
        # Based on global land mean and standard deviation values
        bioclim_means = np.array([
            141,    # bio_1: Annual Mean Temperature (°C * 10)
            99,     # bio_2: Mean Diurnal Range (°C * 10)  
            36,     # bio_3: Isothermality (%)
            6675,   # bio_4: Temperature Seasonality (°C * 100)
            299,    # bio_5: Max Temperature of Warmest Month (°C * 10)
            18,     # bio_6: Min Temperature of Coldest Month (°C * 10)
            281,    # bio_7: Temperature Annual Range (°C * 10)
            230,    # bio_8: Mean Temperature of Wettest Quarter (°C * 10)
            100,    # bio_9: Mean Temperature of Driest Quarter (°C * 10)
            230,    # bio_10: Mean Temperature of Warmest Quarter (°C * 10)
            60,     # bio_11: Mean Temperature of Coldest Quarter (°C * 10)
            786,    # bio_12: Annual Precipitation (mm)
            103,    # bio_13: Precipitation of Wettest Month (mm)
            9,      # bio_14: Precipitation of Driest Month (mm)
            54,     # bio_15: Precipitation Seasonality (CV)
            283,    # bio_16: Precipitation of Wettest Quarter (mm)
            42,     # bio_17: Precipitation of Driest Quarter (mm)
            359,    # bio_18: Precipitation of Warmest Quarter (mm)
            359     # bio_19: Precipitation of Coldest Quarter (mm)
        ])
        
        bioclim_stds = np.array([
            110,    # bio_1 std
            45,     # bio_2 std
            15,     # bio_3 std
            3500,   # bio_4 std
            115,    # bio_5 std
            123,    # bio_6 std
            145,    # bio_7 std
            110,    # bio_8 std
            115,    # bio_9 std
            110,    # bio_10 std
            125,    # bio_11 std
            850,    # bio_12 std
            95,     # bio_13 std
            20,     # bio_14 std
            35,     # bio_15 std
            280,    # bio_16 std
            55,     # bio_17 std
            370,    # bio_18 std
            370     # bio_19 std
        ])
        
        # SoilGrids global soil property statistics (approximate)
        # Values in original units before any scaling
        soil_means = np.array([
            1500,   # bdod: Bulk density (cg/cm³)
            150,    # cec: Cation exchange capacity (mmol(c)/kg)
            100,    # cfvo: Coarse fragments (cm³/dm³)
            230,    # clay: Clay content (g/kg)
            20,     # nitrogen: Nitrogen (cg/kg)
            25,     # ocd: Organic carbon density (hg/m³)
            65,     # phh2o: pH in H2O (pH*10)
            420     # sand: Sand content (g/kg)
        ])
        
        soil_stds = np.array([
            300,    # bdod std
            120,    # cec std
            150,    # cfvo std
            150,    # clay std
            25,     # nitrogen std
            30,     # ocd std
            15,     # phh2o std
            280     # sand std
        ])
        
        return {
            'bioclim_means': bioclim_means.astype(np.float32),
            'bioclim_stds': bioclim_stds.astype(np.float32),
            'soil_means': soil_means.astype(np.float32),
            'soil_stds': soil_stds.astype(np.float32)
        }


def download_environmental_data(cache_dir: Optional[Union[str, Path]] = None) -> EnvironmentalDataDownloader:
    """Factory function to create and initialize environmental data downloader."""
    return EnvironmentalDataDownloader(cache_dir)


if __name__ == "__main__":
    # Test the environmental data downloader
    print("Testing Environmental Data Downloader...")
    
    downloader = download_environmental_data()
    
    # Test data extraction for a few locations
    test_locations = [
        (40.7128, -74.0060),  # New York
        (34.0522, -118.2437), # Los Angeles
        (51.5074, -0.1278)    # London
    ]
    
    lats, lons = zip(*test_locations)
    
    try:
        env_data = downloader.extract_environmental_data(list(lats), list(lons))
        
        print(f"✓ Successfully extracted environmental data for {len(env_data)} locations")
        
        for i, data in enumerate(env_data):
            print(f"  Location {i+1} ({data['coordinates']}):")
            print(f"    Bioclim shape: {data['bioclim'].shape}")
            print(f"    Soil shape: {data['soil'].shape}")
        
        # Test stats
        stats = downloader.get_environmental_stats()
        print(f"✓ Environmental stats available:")
        print(f"  Bioclim means: {len(stats['bioclim_means'])} variables")
        print(f"  Soil means: {len(stats['soil_means'])} variables")
        
    except Exception as e:
        print(f"✗ Environmental data extraction failed: {e}")
        import traceback
        traceback.print_exc()
