#!/usr/bin/env python3
"""
Demonstration of Satellite Embeddings Extractor

This script demonstrates how to use the satellite embeddings extraction system
to extract features from geospatial foundation models.
"""

from datetime import datetime
import numpy as np
from sat_embeddings_extractor import (
    SatelliteEmbeddingsExtractor, 
    ExtractionRequest, 
    ModelType,
    extract_embeddings
)

def demo_basic_usage():
    """Demonstrate basic usage of the extraction system."""
    print("=" * 60)
    print("SATELLITE EMBEDDINGS EXTRACTOR DEMONSTRATION")
    print("=" * 60)
    
    # Example coordinates (New York and Los Angeles)
    latitudes = [40.7128, 34.0522]
    longitudes = [-74.0060, -118.2437]
    datetimes = [datetime(2023, 6, 15), datetime(2023, 6, 15)]
    
    print(f"\nTest locations:")
    for i, (lat, lon, dt) in enumerate(zip(latitudes, longitudes, datetimes)):
        print(f"  {i+1}. Lat: {lat:8.4f}, Lon: {lon:9.4f}, Date: {dt.strftime('%Y-%m-%d')}")
    
    print(f"\nAvailable models:")
    for model in ModelType:
        print(f"  - {model.value}")

def demo_convenience_function():
    """Demonstrate the convenience function."""
    print("\n" + "-" * 40)
    print("TESTING CONVENIENCE FUNCTION")
    print("-" * 40)
    
    # Use the convenience function (this will fail without actual API keys and models)
    try:
        embeddings = extract_embeddings(
            latitudes=[40.7128],
            longitudes=[-74.0060],
            datetimes=[datetime(2023, 6, 15)],
            model="satbird",
            imagery_source="planetary_computer"
        )
        print(f"✓ Successfully extracted embeddings!")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Type: {type(embeddings)}")
        
    except Exception as e:
        print(f"✗ Extraction failed (expected without API keys/models): {str(e)[:100]}...")

def demo_advanced_usage():
    """Demonstrate advanced usage with custom parameters."""
    print("\n" + "-" * 40)  
    print("TESTING ADVANCED USAGE")
    print("-" * 40)
    
    try:
        # Create extractor with custom settings
        extractor = SatelliteEmbeddingsExtractor(
            cache_dir="./custom_cache",
            device="cpu"  # Force CPU for demo
        )
        
        # Create extraction request with custom parameters
        request = ExtractionRequest(
            latitudes=[37.7749],  # San Francisco
            longitudes=[-122.4194],
            datetimes=[datetime(2023, 7, 4)],
            model=ModelType.GALILEO,
            imagery_source="planetary_computer",
            temporal_window_days=15,
            max_cloud_cover=0.2,
            include_environmental=False,
            scale=200.0,
        )
        
        print(f"Created extraction request:")
        print(f"  Model: {request.model.value}")
        print(f"  Imagery source: {request.imagery_source}")
        print(f"  Temporal window: {request.temporal_window_days} days")
        print(f"  Max cloud cover: {request.max_cloud_cover * 100}%")
        print(f"  Pool scale (m): {request.scale}")
        
        # Validate the request
        extractor._validate_request(request)
        print(f"✓ Request validation successful")
        
        # Attempt extraction (will fail without API keys)
        result = extractor.extract_embeddings(request)
        print(f"✓ Successfully extracted embeddings!")
        print(f"  Embeddings shape: {result.embeddings.shape}")
        print(f"  Metadata: {result.metadata}")
        
    except Exception as e:
        print(f"✗ Advanced extraction failed (expected without API keys/models): {str(e)[:100]}...")

def demo_model_wrappers():
    """Demonstrate model wrapper functionality."""
    print("\n" + "-" * 40)
    print("TESTING MODEL WRAPPERS")
    print("-" * 40)
    
    from model_wrappers import create_model_wrapper
    
    models = ["satbird", "galileo", "prithvi_v2"]
    
    for model_name in models:
        try:
            wrapper = create_model_wrapper(model_name, device="cpu")
            print(f"\n{model_name.upper()} Model:")
            print(f"  Expected bands: {wrapper.get_expected_bands()}")
            print(f"  Embedding dimension: {wrapper.get_embedding_dimension()}")
            print(f"  Device: {wrapper.device}")
            print(f"  Loaded: {wrapper.loaded}")
            
        except Exception as e:
            print(f"✗ {model_name} wrapper creation failed: {e}")

def demo_data_preprocessing():
    """Demonstrate data preprocessing functionality."""
    print("\n" + "-" * 40)
    print("TESTING DATA PREPROCESSING")
    print("-" * 40)
    
    from data_preprocessing import create_preprocessor
    
    # Create dummy imagery data
    dummy_imagery = {
        "imagery": [np.random.rand(4, 32, 32).astype(np.float32)],
        "metadata": [{"success": True, "cloud_cover": 5.0}],
        "bands": ["B02", "B03", "B04", "B08"],
        "coordinates": [(40.7128, -74.0060)],
        "datetimes": [datetime.now()]
    }
    
    preprocessor = create_preprocessor()
    
    for model_type in ["satbird", "prithvi_v2"]:
        try:
            if model_type == "prithvi_v2":
                # Modify dummy data for Prithvi (needs more bands)
                test_imagery = dummy_imagery.copy()
                test_imagery["imagery"] = [np.random.rand(10, 32, 32).astype(np.float32)]
                test_imagery["bands"] = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
            else:
                test_imagery = dummy_imagery
                
            result = preprocessor.preprocess(test_imagery, model_type)
            print(f"\n{model_type.upper()} Preprocessing:")
            print(f"  Input shape: {test_imagery['imagery'][0].shape}")
            print(f"  Output shape: {result['imagery'].shape}")
            print(f"  Output type: {type(result['imagery'])}")
            
        except Exception as e:
            print(f"✗ {model_type} preprocessing failed: {e}")

def demo_imagery_downloaders():
    """Demonstrate imagery downloader functionality."""
    print("\n" + "-" * 40)
    print("TESTING IMAGERY DOWNLOADERS")
    print("-" * 40)
    
    from imagery_download import create_downloader
    
    sources = ["planetary_computer", "earth_engine"]
    
    for source in sources:
        try:
            downloader = create_downloader(source)
            print(f"\n{source.upper()} Downloader:")
            print(f"  Type: {type(downloader).__name__}")
            print(f"  Cache dir: {downloader.cache_dir}")
            
            if hasattr(downloader, 'get_bands_for_model'):
                print(f"  SatBird bands: {downloader.get_bands_for_model('satbird')}")
                
        except Exception as e:
            print(f"✗ {source} downloader creation failed: {str(e)[:100]}...")

def main():
    """Run all demonstrations."""
    print("Starting Satellite Embeddings Extractor Demonstration...")
    
    try:
        demo_basic_usage()
        demo_convenience_function()
        demo_advanced_usage()
        demo_model_wrappers()
        demo_data_preprocessing()
        demo_imagery_downloaders()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nNOTE: Many functions will fail without proper API keys and model weights.")
        print("This is expected behavior for the demonstration.")
        print("\nTo use the system with real data:")
        print("1. Set up Microsoft Planetary Computer or Google Earth Engine API keys")
        print("2. Download/configure model weights for SatBird, Galileo, and Prithvi")
        print("3. Install the package (e.g., pip install . or pip install -e .[dev])")
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
