"""
Satellite Embeddings Extractor

This module provides a unified interface for extracting geospatial foundation model 
representations for points in space and time using SatBird, Galileo, and Prithvi V2 models.
"""

from typing import List, Union, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import torch
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import warnings
from pathlib import Path

from geogist.paths import get_cache_path


class ModelType(Enum):
    """Supported foundation models."""
    SATBIRD = "satbird"
    GALILEO = "galileo" 
    PRITHVI_V2 = "prithvi_v2"
    TESSERA = "tessera"
    ALPHAEARTH = "alphaearth"
    TAXABIND = "taxabind"
    DINOV2 = "dinov2"
    DINOV3 = "dinov3"


@dataclass
class ExtractionRequest:
    """Request configuration for embedding extraction."""
    latitudes: List[float]
    longitudes: List[float]
    datetimes: List[datetime]
    model: ModelType
    imagery_source: str = "planetary_computer"  # "planetary_computer", "earth_engine", or "sentinel_cloudless"
    temporal_window_days: int = 30  # Days around target date to search for imagery
    max_cloud_cover: float = 0.1  # Maximum cloud cover threshold
    include_environmental: bool = True  # Whether to include environmental variables (for SatBird)
    scale: Optional[float] = None  # Optional pooling scale in meters
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result of embedding extraction."""
    embeddings: np.ndarray  # Shape: [num_locations, embedding_dim]
    metadata: Dict[str, Any]  # Additional metadata about the extraction


class SatelliteEmbeddingsExtractor:
    """
    Main class for extracting satellite-based embeddings from geospatial foundation models.
    
    This class provides a unified interface to download satellite imagery and extract
    feature representations using SatBird, Galileo, or Prithvi V2 models.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        device: str = "auto",
    ):
        """
        Initialize the extractor.
        
        Args:
            cache_dir: Directory to cache downloaded imagery and models. Defaults
                to a temporary directory if not provided.
            device: Device to run models on ("auto", "cpu", "cuda")
        """
        self.cache_dir = str(get_cache_path(cache_dir))
        self.device = self._setup_device(device)
        self._models = {}  # Cache for loaded models
        
    def extract_embeddings(self, request: ExtractionRequest) -> EmbeddingResult:
        """
        Extract embeddings for the given locations and times.
        
        Args:
            request: ExtractionRequest containing coordinates, times, and model config
            
        Returns:
            EmbeddingResult containing embeddings and metadata
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If imagery download or model inference fails
        """
        # Validate inputs
        self._validate_request(request)
        
        # Download satellite imagery
        imagery_data = self._download_imagery(request)
        
        # Load model if not cached
        if request.model not in self._models:
            self._models[request.model] = self._load_model(
                request.model, request.model_kwargs
            )
            
        # Preprocess data for the specific model
        processed_data = self._preprocess_data(imagery_data, request)
        
        # Extract embeddings
        embeddings = self._extract_model_embeddings(
            processed_data,
            self._models[request.model],
            request.model,
            request.scale,
        )
        
        # Prepare metadata
        metadata = {
            "model": request.model.value,
            "imagery_source": request.imagery_source,
            "num_locations": len(request.latitudes),
            "temporal_window_days": request.temporal_window_days,
            "max_cloud_cover": request.max_cloud_cover,
            "include_environmental": request.include_environmental,
            "scale_meters": request.scale,
            "model_kwargs": dict(request.model_kwargs),
        }
        
        return EmbeddingResult(embeddings=embeddings, metadata=metadata)
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _validate_request(self, request: ExtractionRequest) -> None:
        """Validate the extraction request."""
        if not (len(request.latitudes) == len(request.longitudes) == len(request.datetimes)):
            raise ValueError("Latitudes, longitudes, and datetimes must have the same length")
        
        if not request.latitudes:
            raise ValueError("At least one location must be provided")
            
        for lat in request.latitudes:
            if not -90 <= lat <= 90:
                raise ValueError(f"Invalid latitude: {lat}")
                
        for lon in request.longitudes:
            if not -180 <= lon <= 180:
                raise ValueError(f"Invalid longitude: {lon}")
                
        cloudless_models = {
            ModelType.TAXABIND,
            ModelType.DINOV2,
            ModelType.DINOV3,
        }
        cloudless_source = "sentinel_cloudless"

        if request.model in cloudless_models and request.imagery_source != cloudless_source:
            warnings.warn(
                f"Model '{request.model.value}' requires the Sentinel-2 cloudless imagery source. "
                f"Overriding imagery_source='{request.imagery_source}' with '{cloudless_source}'.",
                UserWarning,
            )
            request.imagery_source = cloudless_source

        if request.imagery_source == cloudless_source and request.model not in cloudless_models:
            raise ValueError(
                f"imagery_source '{cloudless_source}' is only supported for "
                "TaxaBind, DINOv2, and DINOv3 models."
            )

        # TESSERA and AlphaEarth use precomputed embeddings and do not require imagery sources
        if request.model not in (ModelType.TESSERA, ModelType.ALPHAEARTH):
            allowed_sources = ["planetary_computer", "earth_engine", cloudless_source]
            if request.imagery_source not in allowed_sources:
                raise ValueError(
                    "imagery_source must be 'planetary_computer', 'earth_engine', "
                    "or 'sentinel_cloudless'"
                )

        if request.scale is not None:
            if not np.isfinite(request.scale):
                raise ValueError("scale must be finite when provided")
            if request.scale <= 0:
                raise ValueError("scale must be positive when provided")
    
    def _download_imagery(self, request: ExtractionRequest) -> Dict[str, Any]:
        """Download satellite imagery for the requested locations and times."""
        # TESSERA and AlphaEarth use pre-computed embeddings
        if request.model in (ModelType.TESSERA, ModelType.ALPHAEARTH):
            return {
                "coordinates": list(zip(request.latitudes, request.longitudes)),
                "datetimes": request.datetimes,
                "metadata": {
                    "source": request.model.value,
                    "skip_download": True
                }
            }
        
        from .imagery_download import create_downloader
        from .model_wrappers import create_model_wrapper
        
        # Create appropriate downloader
        downloader = create_downloader(request.imagery_source, cache_dir=self.cache_dir)
        
        # Get expected bands for the model
        model_wrapper = create_model_wrapper(request.model.value, **(request.model_kwargs or {}))
        bands = model_wrapper.get_expected_bands()
        image_size = model_wrapper.get_expected_image_size()
        
        # Download imagery
        imagery_data = downloader.download_for_locations(
            latitudes=request.latitudes,
            longitudes=request.longitudes,
            datetimes=request.datetimes,
            bands=bands,
            image_size=image_size,
            temporal_window_days=request.temporal_window_days,
            max_cloud_cover=request.max_cloud_cover
        )
        
        return imagery_data
    
    def _load_model(
        self,
        model_type: ModelType,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load the specified model."""
        from .model_wrappers import create_model_wrapper
        
        # Create and load model wrapper
        wrapper_kwargs = dict(model_kwargs or {})
        if model_type == ModelType.TESSERA:
            wrapper_kwargs.setdefault("cache_dir", self.cache_dir)

        model_wrapper = create_model_wrapper(
            model_type.value,
            device=self.device,
            **wrapper_kwargs,
        )
        model_wrapper.load_model()
        
        return model_wrapper
    
    def _preprocess_data(self, imagery_data: Dict[str, Any], request: ExtractionRequest) -> Any:
        """Preprocess imagery data for the specific model."""
        from .data_preprocessing import create_preprocessor
        
        # Create preprocessor and process data
        preprocessor = create_preprocessor(cache_dir=self.cache_dir)
        processed_data = preprocessor.preprocess(imagery_data, request.model.value)
        
        return processed_data
    
    def _extract_model_embeddings(
        self,
        data: Any,
        model: Any,
        model_type: ModelType,
        scale: Optional[float],
    ) -> np.ndarray:
        """Extract embeddings using the specified model."""
        # Extract embeddings using the model wrapper
        embeddings = model.extract_embeddings(data, scale=scale)

        return embeddings


@lru_cache(maxsize=None)
def get_extractor(
    cache_dir: Optional[Union[str, Path]] = None,
    device: str = "auto",
) -> SatelliteEmbeddingsExtractor:
    """
    Get an instance of the SatelliteEmbeddingsExtractor.
    
    Args:
        cache_dir: Optional directory for caching imagery and models
        device: Device to run models on
        
    Returns:
        An instance of SatelliteEmbeddingsExtractor
    """
    return SatelliteEmbeddingsExtractor(cache_dir=cache_dir, device=device)


# Convenience function for quick usage
def extract_embeddings(latitudes: List[float],
                      longitudes: List[float], 
                      datetimes: List[datetime],
                      model: str,
                      scale: Optional[float] = None,
                      *,
                      cache_dir: Optional[Union[str, Path]] = None,
                      **kwargs) -> np.ndarray:
    """
    Convenience function to extract embeddings.
    
    Args:
        latitudes: List of latitude coordinates
        longitudes: List of longitude coordinates  
        datetimes: List of datetime objects
        model: Model name ("satbird", "galileo", "prithvi_v2", "tessera",
            "alphaearth", "taxabind", "dinov2", "dinov3")
        scale: Optional side length in meters for pooling the final feature map
        cache_dir: Optional directory for caching imagery and models
        **kwargs: Additional parameters passed to ExtractionRequest
        
    Returns:
        Array of embeddings with shape [num_locations, embedding_dim]
    """
    extractor = get_extractor(cache_dir=cache_dir)
    if "scale" in kwargs:
        if scale is not None:
            raise TypeError("scale specified twice")
        scale = kwargs.pop("scale")

    request = ExtractionRequest(
        latitudes=latitudes,
        longitudes=longitudes, 
        datetimes=datetimes,
        model=ModelType(model),
        scale=scale,
        **kwargs
    )
    result = extractor.extract_embeddings(request)
    return result.embeddings


if __name__ == "__main__":
    # Example usage
    from datetime import datetime
    
    # Example coordinates and times
    lats = [40.7128, 34.0522]  # New York, Los Angeles
    lons = [-74.0060, -118.2437]
    times = [datetime(2023, 6, 15), datetime(2023, 6, 15)]
    
    # Extract embeddings using SatBird
    try:
        embeddings = extract_embeddings(lats, lons, times, "satbird")
        print(f"Extracted embeddings with shape: {embeddings.shape}")
    except NotImplementedError:
        print("Implementation in progress...")
