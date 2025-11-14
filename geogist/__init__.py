"""
GeoGist - A unified interface for geospatial foundation models.

This package provides a unified interface for extracting geospatial foundation model 
representations from satellite imagery. It integrates four major geospatial foundation 
models plus pre-computed embedding sources such as TESSERA and Google AlphaEarth, handling satellite imagery download, 
preprocessing, and feature extraction automatically.

Main components:
- SatelliteEmbeddingsExtractor: Main API for extracting embeddings
- ImageryDownloader: Download satellite imagery from various sources
- ModelWrapper: Wrapper classes for different foundation models
- DataPreprocessor: Preprocessing pipeline for different models

Example usage:
    >>> from geogist import extract_embeddings
    >>> embeddings = extract_embeddings(
    ...     latitudes=[40.7128], 
    ...     longitudes=[-74.0060], 
    ...     datetimes=[datetime(2023, 6, 15)],
    ...     model="satbird"
    ... )
"""

__version__ = "0.1.0"
__author__ = "GeoGist Team"
__email__ = "contact@example.com"

# Import main classes and functions for easy access
from .sat_embeddings_extractor import (
    SatelliteEmbeddingsExtractor,
    ExtractionRequest,
    EmbeddingResult,
    ModelType,
    extract_embeddings,
)

from .imagery_download import (
    ImageryDownloader,
    PlanetaryComputerDownloader,
    EarthEngineDownloader,
    SentinelCloudlessDownloader,
    create_downloader,
)

from .model_wrappers import (
    ModelWrapper,
    SatBirdWrapper,
    GalileoWrapper, 
    PrithviWrapper,
    AlphaEarthWrapper,
    TaxaBindWrapper,
    DinoV2Wrapper,
    DinoV3Wrapper,
    create_model_wrapper,
)

from .data_preprocessing import (
    DataPreprocessor,
    TemporalAggregator,
)

from .environmental_data import (
    download_environmental_data,
)

__all__ = [
    # Main API
    "SatelliteEmbeddingsExtractor",
    "ExtractionRequest", 
    "EmbeddingResult",
    "ModelType",
    "extract_embeddings",
    
    # Imagery download
    "ImageryDownloader",
    "PlanetaryComputerDownloader", 
    "EarthEngineDownloader",
    "SentinelCloudlessDownloader",
    "create_downloader",
    
    # Model wrappers
    "ModelWrapper",
    "SatBirdWrapper",
    "GalileoWrapper", 
    "PrithviWrapper",
    "AlphaEarthWrapper",
    "TaxaBindWrapper",
    "DinoV2Wrapper",
    "DinoV3Wrapper",
    "create_model_wrapper",
    
    # Data preprocessing
    "DataPreprocessor",
    "TemporalAggregator",
    
    # Environmental data
    "download_environmental_data",
]
