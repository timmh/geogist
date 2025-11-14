"""
Data Preprocessing Module

This module handles preprocessing of satellite imagery for different geospatial foundation models,
including band selection, normalization, resizing, and temporal aggregation.
"""

from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
import sys
import logging
import json
from scipy.interpolate import griddata

from geogist.single_file_galileo import construct_galileo_input
from .paths import get_cache_path


logger = logging.getLogger(__name__)

CLIP_RGB_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_RGB_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMAGENET_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def inpaint_nans(data: np.ndarray) -> np.ndarray:
    """
    Inpaint NaN values in a 2D array using nearest neighbor interpolation.
    
    Args:
        data: 2D numpy array with NaN values to be inpainted
        
    Returns:
        Array with NaN values replaced by interpolated values
    """

    if not np.isnan(data).any():
        return data # No NaNs to inpaint
    
    data = data.copy()
    
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask): # All NaNs
        data[...] = 0
        return data

    valid_coords = np.array(np.nonzero(valid_mask)).T
    valid_values = data[valid_mask]
    
    nan_coords = np.array(np.nonzero(~valid_mask)).T
    
    inpainted_values = griddata(valid_coords, valid_values, nan_coords, method='nearest')
    
    data[~valid_mask] = inpainted_values

    return data


def handle_imagery_nans(imagery: np.ndarray, band_names: List[str]) -> np.ndarray:
    """
    Handle NaN values in satellite imagery bands with smart inpainting strategy.
    
    For each band:
    - If there is at least one non-NaN pixel, inpaint using nearest neighbor interpolation
    - If all pixels are NaN, replace with the mean value for that band from statistics
    
    Args:
        imagery: 3D numpy array with shape (bands, height, width)
        band_names: List of band names corresponding to the bands
        
    Returns:
        Array with NaN values handled appropriately
    """
    if not np.isnan(imagery).any():
        return imagery  # No NaNs to handle
    
    imagery = imagery.copy()
    
    # Get fallback statistics for bands
    stats = get_fallback_statistics()
    
    for band_idx in range(imagery.shape[0]):
        band_data = imagery[band_idx]
        
        if np.isnan(band_data).any():
            valid_mask = ~np.isnan(band_data)
            
            if np.any(valid_mask):
                # At least one non-NaN pixel exists, inpaint
                logger.info(f"Inpainting NaN values in band {band_names[band_idx] if band_idx < len(band_names) else band_idx}")
                imagery[band_idx] = inpaint_nans(band_data)
            else:
                # All pixels are NaN, replace with band mean
                band_name = band_names[band_idx] if band_idx < len(band_names) else f"band_{band_idx}"
                
                # Try to find the band mean from statistics
                band_mean = 0.0  # Default fallback
                for model_stats in stats.values():
                    if 'bands' in model_stats and 'means' in model_stats:
                        if band_name in model_stats['bands']:
                            band_mean_idx = model_stats['bands'].index(band_name)
                            if band_mean_idx < len(model_stats['means']):
                                band_mean = model_stats['means'][band_mean_idx]
                                break
                
                logger.warning(f"All pixels NaN in band {band_name}, replacing with mean value: {band_mean}")
                imagery[band_idx].fill(band_mean)
    
    return imagery


def load_band_statistics(stats_file: str = "band_statistics.json") -> Dict[str, Any]:
    """
    Load precomputed band statistics from JSON file.
    
    Args:
        stats_file: Path to the JSON file containing band statistics
        
    Returns:
        Dictionary containing statistics for each model
    """
    # Try different possible locations for the statistics file
    possible_paths = [
        Path(__file__).parent / stats_file,
        Path(__file__).parent.parent / stats_file,
        Path(stats_file)
    ]
    
    for stats_path in possible_paths:
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                logger.info(f"Loaded band statistics from {stats_path}")
                return stats
            except Exception as e:
                logger.warning(f"Failed to load statistics from {stats_path}: {e}")
                continue
    
    # If no statistics file found, return fallback values
    logger.warning(f"Could not load statistics file {stats_file}. Using fallback values.")
    return get_fallback_statistics()


def get_fallback_statistics() -> Dict[str, Any]:
    """
    Return fallback statistics if the computed statistics file is not available.
    These are literature-based values for Sentinel-2 surface reflectance.
    """
    return {
        'satbird': {
            'bands': ['B02', 'B03', 'B04', 'B08'],
            'means': [1105.0, 1355.0, 1552.0, 2743.0],  # Blue, Green, Red, NIR
            'stds': [1809.0, 1757.0, 1888.0, 1742.0]
        },
        'galileo': {
            'bands': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
            'means': [1395.34, 1395.34, 1338.40, 1343.10, 1543.86, 2186.20, 2525.09, 2410.34, 2750.29, 2750.29, 2234.91, 2234.91, 1474.53],
            'stds': [917.70, 917.70, 913.30, 1092.68, 1047.22, 1048.01, 1143.69, 1098.98, 1204.47, 1204.47, 1145.98, 1145.98, 980.24]
        },
        'prithvi': {
            'bands': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
            'means': [1087.0, 1342.0, 1433.0, 1363.0, 1363.0, 1363.0],
            'stds': [2248.0, 2179.0, 2178.0, 1049.0, 1049.0, 1049.0]
        }
    }


# Load statistics at module import time
BAND_STATISTICS = load_band_statistics()


class DataPreprocessor:
    """Base class for data preprocessing."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache_dir = get_cache_path(cache_dir)
    
    def preprocess(self, imagery_data: Dict[str, Any], model_type: str, **kwargs) -> Dict[str, Any]:
        """
        Preprocess imagery data for a specific model.
        
        Args:
            imagery_data: Raw imagery data from downloaders
            model_type: Type of model ("satbird", "galileo", "prithvi_v2", "tessera")
            **kwargs: Additional model-specific parameters
            
        Returns:
            Preprocessed data ready for model inference
        """
        if model_type.lower() == "satbird":
            include_env = kwargs.get('include_environmental', True)
            return self._preprocess_satbird(imagery_data, include_environmental=include_env)
        elif model_type.lower() == "galileo":
            return self._preprocess_galileo(imagery_data)
        elif model_type.lower() == "prithvi_v2":
            return self._preprocess_prithvi(imagery_data)
        elif model_type.lower() == "tessera":
            return self._preprocess_tessera(imagery_data)
        elif model_type.lower() == "alphaearth":
            return self._preprocess_alphaearth(imagery_data)
        elif model_type.lower() == "taxabind":
            return self._preprocess_taxabind(imagery_data)
        elif model_type.lower() == "dinov2":
            return self._preprocess_dinov2(imagery_data)
        elif model_type.lower() == "dinov3":
            return self._preprocess_dinov3(imagery_data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _preprocess_satbird(self, imagery_data: Dict[str, Any], include_environmental: bool = True) -> Dict[str, Any]:
        """Preprocess data for SatBird model."""
        processed_images = []
        
        # Get SatBird normalization stats from precomputed statistics
        satbird_stats = BAND_STATISTICS.get('satbird', {})
        band_means = np.array(satbird_stats.get('means', [1105.0, 1355.0, 1552.0, 2743.0]))
        band_stds = np.array(satbird_stats.get('stds', [1809.0, 1757.0, 1888.0, 1742.0]))
        
        logger.info(f"Using SatBird normalization - means: {band_means}, stds: {band_stds}")
        
        # Process satellite imagery
        for i, imagery in enumerate(imagery_data["imagery"]):
            if np.isnan(imagery).any():
                # Use smart NaN handling strategy
                imagery = handle_imagery_nans(imagery, imagery_data["bands"])
                logger.info(f"Applied smart NaN handling to imagery for index {i}")
            
            # Continue with processing
            # Ensure we have the expected bands for SatBird (B02, B03, B04, B08)
            expected_bands = ["B02", "B03", "B04", "B08"]
            if len(imagery_data["bands"]) != len(expected_bands):
                # Reorder/select bands
                imagery = self._reorder_bands(imagery, imagery_data["bands"], expected_bands)
            
            # Resize to 64x64
            imagery = self._resize_imagery(imagery, (64, 64))
            
            # Normalize
            sat_data = self._normalize_imagery(imagery, band_means, band_stds)
            
            # Add environmental data if requested
            if include_environmental:
                env_data = self._get_environmental_data(imagery_data, i)
                # Combine satellite and environmental data
                combined_data = self._combine_satbird_data(sat_data, env_data)
                processed_images.append(combined_data)
            else:
                processed_images.append(sat_data)
        
        # Stack into batch
        batch = np.stack(processed_images, axis=0)
        
        return {
            "imagery": torch.from_numpy(batch).float(),
            "metadata": imagery_data["metadata"]
        }
    
    def _preprocess_galileo(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data for Galileo model."""

        processed_data = []
        
        for i, imagery in enumerate(imagery_data["imagery"]):
            if np.isnan(imagery).any():
                # Use smart NaN handling strategy
                imagery = handle_imagery_nans(imagery, imagery_data["bands"])
                logger.info(f"Applied smart NaN handling to Galileo imagery for index {i}")
            
            # Ensure we have expected S2 bands
            s2_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
            if len(imagery_data["bands"]) != len(s2_bands):
                imagery = self._reorder_bands(imagery, imagery_data["bands"], s2_bands)
            
            galileo_stats = BAND_STATISTICS.get('galileo', {})
            band_means = np.array(galileo_stats['means'])
            band_stds = np.array(galileo_stats['stds'])
            imagery = self._normalize_imagery(imagery, band_means, band_stds)

            # Resize imagery to appropriate size for Galileo
            imagery = self._resize_imagery(imagery, (224, 224))
            
            # Convert to tensor and add temporal dimension
            # Galileo expects [H, W, T, D] format
            h, w = imagery.shape[1], imagery.shape[2]
            t = 1  # Single timestep
            imagery_tensor = torch.from_numpy(imagery).float()
            imagery_tensor = imagery_tensor.permute(1, 2, 0).unsqueeze(2)  # [H, W, D] -> [H, W, 1, D]
            
            # Create Galileo input
            try:
                galileo_input = construct_galileo_input(
                    s2=imagery_tensor,
                )
                processed_data.append(galileo_input)
            except Exception as e:
                print(f"Failed to create Galileo input for sample {i}: {e}")
                processed_data.append(self._create_dummy_galileo_input())
        
        # Batch the MaskedOutput objects
        batched_input = self._batch_galileo_inputs(processed_data)
        if batched_input is None:
            fallback = self._create_dummy_galileo_input()
            if fallback is None:
                raise RuntimeError("Unable to construct Galileo inputs for any samples")
            batched_input = {key: value.unsqueeze(0) for key, value in fallback.items()}
        
        return {
            "masked_output": batched_input,
            "metadata": imagery_data["metadata"]
        }
    
    def _preprocess_prithvi(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data for Prithvi V2 model."""
        processed_images = []
        
        # Prithvi V2 normalization constants from Galileo preprocess.py
        # These are the actual values used by Prithvi models for Sentinel-2 bands
        # Order: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
        band_means = BAND_STATISTICS.get('prithvi', {}).get('means',
            np.array([1087.0, 1342.0, 1433.0, 1363.0, 1363.0, 1363.0])
        )
        # Standard deviations for each band
        band_stds = BAND_STATISTICS.get('prithvi', {}).get('stds',
            np.array([2248.0, 2179.0, 2178.0, 1049.0, 1049.0, 1049.0])
        )
        
        for i, imagery in enumerate(imagery_data["imagery"]):
            if np.isnan(imagery).any():
                # Use smart NaN handling strategy
                imagery = handle_imagery_nans(imagery, imagery_data["bands"])
                logger.info(f"Applied smart NaN handling to Prithvi imagery for index {i}")
            
            # Ensure we have the expected bands for Prithvi
            expected_bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
            if len(imagery_data["bands"]) != len(expected_bands):
                imagery = self._reorder_bands(imagery, imagery_data["bands"], expected_bands)
            
            # Resize to 224x224 (typical for Prithvi)
            imagery = self._resize_imagery(imagery, (224, 224))
            
            # Normalize
            imagery = self._normalize_imagery(imagery, band_means, band_stds)
            
            processed_images.append(imagery)
        
        # Stack into batch
        batch = np.stack(processed_images, axis=0)
        
        return {
            "imagery": torch.from_numpy(batch).float(),
            "metadata": imagery_data["metadata"]
        }
    
    def _prepare_coordinate_payload(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        coordinates = imagery_data.get("coordinates", [])
        if coordinates:
            latitudes = [coord[0] for coord in coordinates]
            longitudes = [coord[1] for coord in coordinates]
        else:
            latitudes = [0]
            longitudes = [0]
            
        return {
            "latitudes": latitudes,
            "longitudes": longitudes, 
            "datetimes": imagery_data.get("datetimes", []),
            "metadata": imagery_data.get("metadata", {})
        }

    def _preprocess_tessera(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data for TESSERA model.
        
        TESSERA doesn't require imagery preprocessing since it uses pre-computed embeddings.
        Instead, we pass through the coordinate and datetime information needed for fetching.
        """
        return self._prepare_coordinate_payload(imagery_data)

    def _preprocess_alphaearth(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for the Google AlphaEarth embeddings model."""
        return self._prepare_coordinate_payload(imagery_data)

    def _preprocess_taxabind(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare RGB patches for the TaxaBind encoder."""
        return self._preprocess_rgb_model(
            imagery_data,
            target_size=(224, 224),
            means=CLIP_RGB_MEAN,
            stds=CLIP_RGB_STD,
        )

    def _preprocess_dinov2(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare RGB patches for DINOv2."""
        return self._preprocess_rgb_model(
            imagery_data,
            target_size=(224, 224),
            means=IMAGENET_RGB_MEAN,
            stds=IMAGENET_RGB_STD,
        )

    def _preprocess_dinov3(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare RGB patches for DINOv3."""
        return self._preprocess_rgb_model(
            imagery_data,
            target_size=(224, 224),
            means=IMAGENET_RGB_MEAN,
            stds=IMAGENET_RGB_STD,
        )

    def _preprocess_rgb_model(
        self,
        imagery_data: Dict[str, Any],
        target_size: Tuple[int, int],
        means: np.ndarray,
        stds: np.ndarray,
    ) -> Dict[str, Any]:
        processed_images: List[np.ndarray] = []
        for imagery in imagery_data["imagery"]:
            if imagery is None or not np.isfinite(imagery).any():
                rgb_data = np.zeros((3, target_size[0], target_size[1]), dtype=np.float32)
            else:
                rgb_data = imagery[:3].astype(np.float32)
                rgb_data = self._resize_imagery(rgb_data, target_size)
                np.clip(rgb_data, 0.0, 1.0, out=rgb_data)
            normalized = self._normalize_rgb(rgb_data, means, stds)
            processed_images.append(normalized)

        batch = torch.from_numpy(np.stack(processed_images, axis=0)).float()
        return {
            "imagery": batch,
            "metadata": imagery_data["metadata"],
        }
    
    def _reorder_bands(self, imagery: np.ndarray, current_bands: List[str], target_bands: List[str]) -> np.ndarray:
        """Reorder bands to match target ordering."""
        if len(current_bands) == len(target_bands) and current_bands == target_bands:
            return imagery
        
        # Create mapping from current to target bands
        band_indices = []
        for target_band in target_bands:
            if target_band in current_bands:
                band_indices.append(current_bands.index(target_band))
            else:
                # Handle missing bands by creating zeros or duplicating similar bands
                if target_band == "B05" and "B04" in current_bands:
                    band_indices.append(current_bands.index("B04"))  # Use red as proxy
                elif target_band == "B8A" and "B08" in current_bands:
                    band_indices.append(current_bands.index("B08"))  # Use NIR as proxy
                else:
                    band_indices.append(0)  # Default to first band
        
        return imagery[band_indices]
    
    def _resize_imagery(self, imagery: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize imagery to target size."""
        if imagery.shape[1:] == target_size:
            return imagery
        
        # Convert to tensor for resizing
        tensor = torch.from_numpy(imagery).float().unsqueeze(0)  # Add batch dim
        resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        return resized.squeeze(0).numpy()
    
    def _normalize_imagery(self, imagery: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Normalize imagery using provided means and stds."""
        # Ensure means and stds match number of bands
        if len(means) != imagery.shape[0]:
            means = means[:imagery.shape[0]]
        if len(stds) != imagery.shape[0]:
            stds = stds[:imagery.shape[0]]
        
        # Normalize each band
        normalized = imagery.copy().astype(np.float32)
        for i in range(imagery.shape[0]):
            normalized[i] = (normalized[i] - means[i]) / stds[i]
        
        return normalized

    def _normalize_rgb(self, imagery: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Normalize RGB imagery using channel means and stds."""
        normalized = imagery.copy()
        for channel in range(min(3, normalized.shape[0])):
            normalized[channel] = (normalized[channel] - means[channel]) / stds[channel]
        return normalized
    
    def _create_dummy_galileo_input(self) -> Any:
        """Create dummy Galileo input for missing data."""
        
        # Create minimal S2 tensor
        h, w, t = 4, 4, 1
        s2_tensor = torch.zeros(h, w, t, 10)  # 10 S2 bands
        
        try:
            return construct_galileo_input(s2=s2_tensor, normalize=True)
        except Exception:
            return None
    
    def _batch_galileo_inputs(self, inputs: List[Any]) -> Any:
        """Batch Galileo MaskedOutput objects."""

        if not inputs:
            return None

        batched_inputs = []
        for idx, inp in enumerate(inputs):
            if inp is None:
                inp = self._create_dummy_galileo_input()
                if inp is None:
                    raise RuntimeError(
                        f"Failed to create fallback Galileo input for sample {idx}"
                    )
            batched_inputs.append(inp)

        reference = batched_inputs[0]
        batched: Dict[str, torch.Tensor] = {}
        for key, ref_value in reference.items():
            tensors = []
            for sample_idx, sample in enumerate(batched_inputs):
                if key not in sample:
                    raise KeyError(f"Missing key '{key}' in Galileo input {sample_idx}")
                tensors.append(sample[key])
            try:
                batched[key] = torch.stack(tensors, dim=0)
            except Exception as exc:  # pragma: no cover - defensive path
                raise RuntimeError(f"Failed to batch Galileo inputs for key '{key}': {exc}") from exc

        return batched
    
    def _get_environmental_data(self, imagery_data: Dict[str, Any], location_idx: int) -> Dict[str, np.ndarray]:
        """Get environmental data for a specific location."""
        try:
            from .environmental_data import download_environmental_data
            
            # Get coordinates
            if 'coordinates' in imagery_data and location_idx < len(imagery_data['coordinates']):
                lat, lon = imagery_data['coordinates'][location_idx]
            else:
                # Fallback: use dummy coordinates
                lat, lon = 40.0, -100.0
            
            # Download environmental data
            env_downloader = download_environmental_data(cache_dir=self.cache_dir)
            env_data = env_downloader.extract_environmental_data([lat], [lon])
            
            if env_data:
                return {
                    'bioclim': env_data[0]['bioclim'],
                    'soil': env_data[0]['soil']
                }
            else:
                return self._create_dummy_environmental_data()
                
        except Exception as e:
            print(f"Warning: Failed to get environmental data: {e}")
            return self._create_dummy_environmental_data()
    
    def _create_dummy_environmental_data(self) -> Dict[str, np.ndarray]:
        """Create dummy environmental data for fallback."""
        return {
            'bioclim': np.random.rand(19, 64, 64).astype(np.float32),
            'soil': np.random.rand(8, 64, 64).astype(np.float32)
        }
    
    def _combine_satbird_data(self, sat_data: np.ndarray, env_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine satellite and environmental data for SatBird."""
        # Resize environmental data to match satellite data size
        bioclim_resized = self._resize_environmental_data(env_data['bioclim'], (64, 64))
        soil_resized = self._resize_environmental_data(env_data['soil'], (64, 64))
        
        # Handle NaN values in environmental data
        if np.isnan(bioclim_resized).any():
            bioclim_resized = inpaint_nans(bioclim_resized)
            logger.warning("Found NaN values in bioclim data, inpainted.")
        
        if np.isnan(soil_resized).any():
            for i in range(soil_resized.shape[0]):
                soil_resized[i] = inpaint_nans(soil_resized[i])

            logger.warning("Found NaN values in soil data, inpainted.")
        
        # Normalize environmental data
        from .environmental_data import download_environmental_data
        try:
            env_downloader = download_environmental_data(cache_dir=self.cache_dir)
            stats = env_downloader.get_environmental_stats()
            
            # Normalize bioclim data
            for i in range(bioclim_resized.shape[0]):
                if i < len(stats['bioclim_means']):
                    bioclim_resized[i] = (bioclim_resized[i] - stats['bioclim_means'][i]) / stats['bioclim_stds'][i]
            
            # Normalize soil data  
            for i in range(soil_resized.shape[0]):
                if i < len(stats['soil_means']):
                    soil_resized[i] = (soil_resized[i] - stats['soil_means'][i]) / stats['soil_stds'][i]
                    
        except Exception:
            # Fallback: simple normalization
            bioclim_resized = (bioclim_resized - bioclim_resized.mean()) / (bioclim_resized.std() + 1e-8)
            soil_resized = (soil_resized - soil_resized.mean()) / (soil_resized.std() + 1e-8)
        
        # Final NaN check after normalization
        if np.isnan(bioclim_resized).any():
            bioclim_resized = np.nan_to_num(bioclim_resized, nan=0.0)
            logger.warning("Found NaN values in bioclim data after normalization, filled with zeros")
        
        if np.isnan(soil_resized).any():
            soil_resized = np.nan_to_num(soil_resized, nan=0.0)
            logger.warning("Found NaN values in soil data after normalization, filled with zeros")
        
        # Concatenate satellite + bioclim + soil data
        combined = np.concatenate([sat_data, bioclim_resized, soil_resized], axis=0)
        
        # Final check on combined data
        if np.isnan(combined).any():
            combined = np.nan_to_num(combined, nan=0.0)
            logger.warning("Found NaN values in combined data, filled with zeros")
        
        return combined
    
    def _resize_environmental_data(self, env_data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize environmental data to target size."""
        if env_data.shape[1:] == target_size:
            return env_data
        
        # Convert to tensor for resizing
        tensor = torch.from_numpy(env_data).float().unsqueeze(0)  # Add batch dim
        resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        return resized.squeeze(0).numpy()


class TemporalAggregator:
    """Handles temporal aggregation of multi-temporal imagery."""
    
    def __init__(self, method: str = "median"):
        """
        Initialize temporal aggregator.
        
        Args:
            method: Aggregation method ("median", "mean", "most_recent", "cloud_free")
        """
        self.method = method
    
    def aggregate(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate temporal imagery data.
        
        Args:
            imagery_data: Dictionary containing temporal imagery data
            
        Returns:
            Aggregated imagery data
        """
        if self.method == "median":
            return self._median_aggregate(imagery_data)
        elif self.method == "mean":
            return self._mean_aggregate(imagery_data)
        elif self.method == "most_recent":
            return self._most_recent_aggregate(imagery_data)
        elif self.method == "cloud_free":
            return self._cloud_free_aggregate(imagery_data)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
    
    def _median_aggregate(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate using median values."""
        # Implementation would depend on temporal data structure
        # For now, return as-is
        return imagery_data
    
    def _mean_aggregate(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate using mean values."""
        # Implementation would depend on temporal data structure
        return imagery_data
    
    def _most_recent_aggregate(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select most recent imagery."""
        # Implementation would depend on temporal data structure
        return imagery_data
    
    def _cloud_free_aggregate(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select cloud-free imagery based on metadata."""
        # Implementation would use cloud cover metadata
        return imagery_data


def create_preprocessor(**kwargs) -> DataPreprocessor:
    """Factory function to create data preprocessor."""
    return DataPreprocessor(**kwargs)


if __name__ == "__main__":
    # Test preprocessing
    print("Testing data preprocessing...")
    
    # Create dummy imagery data
    dummy_imagery = {
        "imagery": [np.random.rand(4, 32, 32)],  # Single image with 4 bands
        "metadata": [{"success": True}],
        "bands": ["B02", "B03", "B04", "B08"],
        "coordinates": [(40.7128, -74.0060)],
        "datetimes": [datetime.now()]
    }
    
    preprocessor = create_preprocessor()
    
    # Test SatBird preprocessing
    try:
        satbird_data = preprocessor.preprocess(dummy_imagery, "satbird")
        print(f"SatBird preprocessing successful. Shape: {satbird_data['imagery'].shape}")
    except Exception as e:
        print(f"SatBird preprocessing failed: {e}")
    
    # Test Galileo preprocessing
    try:
        galileo_data = preprocessor.preprocess(dummy_imagery, "galileo")
        print(f"Galileo preprocessing successful.")
    except Exception as e:
        print(f"Galileo preprocessing failed: {e}")
    
    # Test Prithvi preprocessing
    try:
        prithvi_data = preprocessor.preprocess(dummy_imagery, "prithvi_v2")
        print(f"Prithvi preprocessing successful. Shape: {prithvi_data['imagery'].shape}")
    except Exception as e:
        print(f"Prithvi preprocessing failed: {e}")
