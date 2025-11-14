#!/usr/bin/env python3
"""
Compute Band Statistics for Normalization

This script samples 100 random locations within the contiguous United States,
downloads satellite imagery for each location, and computes mean and standard
deviation statistics for different band combinations used by SatBird, Galileo,
and Prithvi models.

The computed statistics are saved to a JSON file for use during preprocessing.
"""

import json
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm

from geogist.imagery_download import create_downloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Contiguous US bounding box (excluding Alaska and Hawaii)
CONUS_BOUNDS = {
    'min_lat': 24.396308,  # Southern tip of Florida Keys
    'max_lat': 49.384358,  # Northern border with Canada
    'min_lon': -125.0,     # West coast
    'max_lon': -66.93457   # East coast
}

# Band configurations for different models
MODEL_BANDS = {
    'satbird': ['B02', 'B03', 'B04', 'B08'],  # Blue, Green, Red, NIR
    'galileo': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],  # Full S2 suite
    'prithvi': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']   # Full S2 suite
}

# Native image sizes for different models (as specified by user)
MODEL_IMAGE_SIZES = {
    'satbird': (64, 64),    # SatBird uses 64x64
    'galileo': (224, 224),  # Galileo uses 224x224 
    'prithvi': (224, 224)   # Prithvi uses 224x224
}

def generate_random_conus_locations(n_locations: int = 10, seed: int = 42) -> List[Tuple[float, float]]:
    """Generate random lat/lon coordinates within the contiguous United States."""
    random.seed(seed)
    np.random.seed(seed)
    
    locations = []
    for _ in range(n_locations):
        lat = random.uniform(CONUS_BOUNDS['min_lat'], CONUS_BOUNDS['max_lat'])
        lon = random.uniform(CONUS_BOUNDS['min_lon'], CONUS_BOUNDS['max_lon'])
        locations.append((lat, lon))
    
    logger.info(f"Generated {len(locations)} random locations within CONUS")
    return locations

def generate_random_dates(n_dates: int = 10, 
                         start_year: int = 2020, 
                         end_year: int = 2023,
                         seed: int = 42) -> List[datetime]:
    """Generate random dates for imagery sampling."""
    random.seed(seed)
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    time_between = end_date - start_date
    days_between = time_between.days
    
    dates = []
    for _ in range(n_dates):
        random_days = random.randint(0, days_between)
        random_date = start_date + timedelta(days=random_days)
        dates.append(random_date)
    
    logger.info(f"Generated {len(dates)} random dates between {start_year} and {end_year}")
    return dates

def download_imagery_sample(locations: List[Tuple[float, float]], 
                           dates: List[datetime],
                           bands: List[str],
                           model_name: str,
                           imagery_source: str = "planetary_computer",
                           max_cloud_cover: float = 0.3,
                           temporal_window_days: int = 30) -> List[np.ndarray]:
    """Download imagery for the given locations and extract band data."""
    
    # Get model-specific image size
    image_size = MODEL_IMAGE_SIZES.get(model_name, (64, 64))
    logger.info(f"Downloading imagery for {len(locations)} locations using {imagery_source} at size {image_size}")
    
    # Create downloader
    downloader = create_downloader(imagery_source)
    
    # Extract coordinates
    lats = [loc[0] for loc in locations]
    lons = [loc[1] for loc in locations]
    
    successful_downloads = []
    failed_downloads = 0
    
    # Download imagery in batches to avoid overwhelming the service
    batch_size = 10
    for i in tqdm(range(0, len(locations), batch_size), desc="Downloading imagery"):
        batch_end = min(i + batch_size, len(locations))
        batch_lats = lats[i:batch_end]
        batch_lons = lons[i:batch_end]
        batch_dates = dates[i:batch_end]
        
        try:
            imagery_data = downloader.download_for_locations(
                latitudes=batch_lats,
                longitudes=batch_lons,
                datetimes=batch_dates,
                bands=bands,
                image_size=image_size,  # Model-specific size
                temporal_window_days=temporal_window_days,
                max_cloud_cover=max_cloud_cover
            )
            
            # Extract successful imagery arrays
            for j, imagery in enumerate(imagery_data["imagery"]):
                if imagery is not None and not np.isnan(imagery).any():
                    # Validate shape consistency using model-specific image size
                    expected_shape = (len(bands), image_size[0], image_size[1])
                    if imagery.shape == expected_shape:
                        successful_downloads.append(imagery)
                    else:
                        failed_downloads += 1
                        logger.warning(f"Invalid shape for location {i+j}: got {imagery.shape}, expected {expected_shape}")
                else:
                    failed_downloads += 1
                    logger.warning(f"Failed to download valid imagery for location {i+j}")
                    
        except Exception as e:
            logger.error(f"Error downloading batch {i//batch_size}: {e}")
            failed_downloads += batch_end - i
    
    logger.info(f"Successfully downloaded {len(successful_downloads)} images, {failed_downloads} failed")
    return successful_downloads

def compute_band_statistics(imagery_arrays: List[np.ndarray], 
                           bands: List[str],
                           image_size: Tuple[int, int] = (64, 64)) -> Dict[str, Any]:
    """Compute mean and standard deviation for each band across all imagery."""
    
    if not imagery_arrays:
        raise ValueError("No valid imagery arrays provided")
    
    logger.info(f"Computing statistics for {len(bands)} bands across {len(imagery_arrays)} images")
    
    # Validate all arrays have the same shape
    expected_shape = (len(bands), image_size[0], image_size[1])
    valid_arrays = []
    for i, array in enumerate(imagery_arrays):
        if array.shape == expected_shape:
            valid_arrays.append(array)
        else:
            logger.warning(f"Skipping array {i} with invalid shape: {array.shape}")
    
    if not valid_arrays:
        raise ValueError("No arrays with valid shape found")
    
    logger.info(f"Using {len(valid_arrays)} valid arrays out of {len(imagery_arrays)} total")
    
    # Stack all imagery into a single array: [n_images, n_bands, height, width]
    try:
        stacked_imagery = np.stack(valid_arrays, axis=0)
    except ValueError as e:
        logger.error(f"Failed to stack arrays: {e}")
        # Try to diagnose the issue
        shapes = [arr.shape for arr in valid_arrays[:5]]  # Show first 5 shapes
        logger.error(f"Sample shapes: {shapes}")
        raise
    
    # Compute statistics across spatial dimensions and images: [n_images, n_bands, h, w] -> [n_bands]
    band_means = np.mean(stacked_imagery, axis=(0, 2, 3))
    band_stds = np.std(stacked_imagery, axis=(0, 2, 3))
    
    # Create statistics dictionary
    stats = {
        'bands': bands,
        'means': band_means.tolist(),
        'stds': band_stds.tolist(),
        'n_samples': len(valid_arrays),
        'n_total_attempted': len(imagery_arrays),
        'computed_date': datetime.now().isoformat()
    }
    
    # Log statistics
    logger.info("Computed band statistics:")
    for i, band in enumerate(bands):
        logger.info(f"  {band}: mean={band_means[i]:.2f}, std={band_stds[i]:.2f}")
    
    return stats

def compute_all_model_statistics(n_locations: int = 50,
                                imagery_source: str = "planetary_computer",
                                max_cloud_cover: float = 0.3,
                                seed: int = 42) -> Dict[str, Any]:
    """Compute statistics for all model band configurations."""
    
    logger.info("Starting computation of band statistics for all models")
    
    # Generate random locations and dates
    locations = generate_random_conus_locations(n_locations, seed)
    dates = generate_random_dates(n_locations, seed=seed)
    
    all_stats = {}
    
    # Compute statistics for each model
    for model_name, bands in MODEL_BANDS.items():
        logger.info(f"\n--- Computing statistics for {model_name.upper()} model ---")
        
        try:
            # Download imagery for this band configuration
            imagery_arrays = download_imagery_sample(
                locations=locations,
                dates=dates,
                bands=bands,
                model_name=model_name,
                imagery_source=imagery_source,
                max_cloud_cover=max_cloud_cover
            )
            
            if len(imagery_arrays) < 5:  # Require at least 5 successful downloads
                logger.warning(f"Only {len(imagery_arrays)} successful downloads for {model_name}. "
                             f"Statistics may not be reliable. Consider increasing max_cloud_cover or temporal_window_days.")
            
            # Compute statistics
            image_size = MODEL_IMAGE_SIZES.get(model_name, (64, 64))
            stats = compute_band_statistics(imagery_arrays, bands, image_size)
            all_stats[model_name] = stats
            
        except Exception as e:
            logger.error(f"Failed to compute statistics for {model_name}: {e}")
            all_stats[model_name] = {
                'error': str(e),
                'bands': bands,
                'computed_date': datetime.now().isoformat()
            }
    
    # Add metadata
    all_stats['metadata'] = {
        'n_locations_requested': n_locations,
        'imagery_source': imagery_source,
        'max_cloud_cover': max_cloud_cover,
        'conus_bounds': CONUS_BOUNDS,
        'computed_date': datetime.now().isoformat(),
        'seed': seed
    }
    
    return all_stats

def save_statistics(stats: Dict[str, Any], 
                   output_path: str = "geogist/band_statistics.json") -> None:
    """Save computed statistics to JSON file."""
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to {output_file}")

def main():
    """Main function to compute and save band statistics."""
    
    logger.info("Starting band statistics computation script")
    
    try:
        # Compute statistics for all models
        all_stats = compute_all_model_statistics(
            n_locations=50,  # Reduced for reliability 
            imagery_source="planetary_computer",  # Can be changed to "earth_engine"
            max_cloud_cover=0.5,  # Allow more clouds to get more samples
            seed=42
        )
        
        # Save to JSON file
        save_statistics(all_stats)
        
        # Print summary
        logger.info("\n=== STATISTICS COMPUTATION COMPLETE ===")
        for model_name in MODEL_BANDS.keys():
            if model_name in all_stats and 'means' in all_stats[model_name]:
                n_samples = all_stats[model_name]['n_samples']
                logger.info(f"{model_name.upper()}: {n_samples} samples processed")
            else:
                logger.info(f"{model_name.upper()}: FAILED")
                
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise

if __name__ == "__main__":
    main()