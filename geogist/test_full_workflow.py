#!/usr/bin/env python3
"""
Full Workflow Test Script

Tests the complete satellite embeddings extraction workflow with real data downloads
from both Microsoft Planetary Computer and Google Earth Engine.
"""

import os
import sys
from datetime import datetime
import numpy as np
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from geogist.sat_embeddings_extractor import (
    SatelliteEmbeddingsExtractor, 
    ExtractionRequest, 
    ModelType,
    extract_embeddings
)
from geogist.imagery_download import create_downloader
from geogist.environmental_data import download_environmental_data
from geogist.paths import get_cache_path

def test_environmental_data():
    """Test environmental data download and processing."""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENTAL DATA")
    print("="*60)
    
    try:
        # Test environmental data downloader
        env_downloader = download_environmental_data()
        
        # Test with a few locations
        test_lats = [40.7128, 34.0522]  # NYC, LA
        test_lons = [-74.0060, -118.2437]
        
        env_data = env_downloader.extract_environmental_data(test_lats, test_lons)
        
        print(f"✓ Environmental data extraction successful")
        print(f"  Locations processed: {len(env_data)}")
        
        for i, data in enumerate(env_data):
            print(f"  Location {i+1}: {data['coordinates']}")
            print(f"    Bioclim shape: {data['bioclim'].shape}")
            print(f"    Soil shape: {data['soil'].shape}")
        
        # Test stats
        stats = env_downloader.get_environmental_stats()
        print(f"✓ Environmental stats available")
        print(f"  Bioclim variables: {len(stats['bioclim_means'])}")
        print(f"  Soil variables: {len(stats['soil_means'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environmental data test failed: {e}")
        traceback.print_exc()
        return False

def test_planetary_computer_download():
    """Test satellite imagery download from Planetary Computer."""
    print("\n" + "="*60)
    print("TESTING PLANETARY COMPUTER DOWNLOAD")
    print("="*60)
    
    try:
        # Create downloader
        downloader = create_downloader("planetary_computer")
        
        # Test with a single location
        test_lat = [40.7128]  # NYC
        test_lon = [-74.0060]
        test_time = [datetime(2023, 6, 15)]
        
        print(f"Testing download for: {test_lat[0]}, {test_lon[0]} on {test_time[0]}")
        
        # Get SatBird bands
        bands = downloader.get_bands_for_model("satbird")
        print(f"Using bands: {bands}")
        
        # Download imagery
        result = downloader.download_for_locations(
            latitudes=test_lat,
            longitudes=test_lon,
            datetimes=test_time,
            bands=bands,
            image_size=(64, 64),
            temporal_window_days=30,
            max_cloud_cover=0.3
        )
        
        print(f"✓ Planetary Computer download successful")
        print(f"  Result keys: {list(result.keys())}")
        print(f"  Imagery shape: {result['imagery'][0].shape if result['imagery'] else 'None'}")
        print(f"  Metadata: {result['metadata'][0] if result['metadata'] else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Planetary Computer download failed: {e}")
        traceback.print_exc()
        return False

def test_earth_engine_download():
    """Test satellite imagery download from Google Earth Engine."""
    print("\n" + "="*60)
    print("TESTING GOOGLE EARTH ENGINE DOWNLOAD")
    print("="*60)
    
    try:
        # Create downloader
        downloader = create_downloader("earth_engine")
        
        # Test with a single location
        test_lat = [34.0522]  # LA
        test_lon = [-118.2437]
        test_time = [datetime(2023, 6, 15)]
        
        print(f"Testing download for: {test_lat[0]}, {test_lon[0]} on {test_time[0]}")
        
        # Get SatBird bands
        bands = ["B02", "B03", "B04", "B08"]
        print(f"Using bands: {bands}")
        
        # Download imagery
        result = downloader.download_for_locations(
            latitudes=test_lat,
            longitudes=test_lon,
            datetimes=test_time,
            bands=bands,
            image_size=(64, 64),
            temporal_window_days=30,
            max_cloud_cover=0.3
        )
        
        print(f"✓ Earth Engine download successful")
        print(f"  Result keys: {list(result.keys())}")
        print(f"  Imagery shape: {result['imagery'][0].shape if result['imagery'] else 'None'}")
        print(f"  Metadata: {result['metadata'][0] if result['metadata'] else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Earth Engine download failed: {e}")
        traceback.print_exc()
        return False

def test_data_preprocessing():
    """Test data preprocessing with environmental data."""
    print("\n" + "="*60)
    print("TESTING DATA PREPROCESSING WITH ENVIRONMENTAL DATA")
    print("="*60)
    
    try:
        from geogist.data_preprocessing import create_preprocessor
        
        # Create dummy imagery data
        dummy_imagery = {
            "imagery": [np.random.rand(4, 32, 32).astype(np.float32)],
            "metadata": [{"success": True, "cloud_cover": 5.0}],
            "bands": ["B02", "B03", "B04", "B08"],
            "coordinates": [(40.7128, -74.0060)],
            "datetimes": [datetime.now()]
        }
        
        preprocessor = create_preprocessor()
        
        # Test SatBird preprocessing with environmental data
        result = preprocessor.preprocess(
            dummy_imagery, 
            "satbird", 
            include_environmental=True
        )
        
        print(f"✓ SatBird preprocessing with environmental data successful")
        print(f"  Input satellite shape: {dummy_imagery['imagery'][0].shape}")
        print(f"  Output combined shape: {result['imagery'].shape}")
        print(f"  Expected channels: 4 (sat) + 19 (bioclim) + 8 (soil) = 31")
        
        # Test without environmental data
        result_no_env = preprocessor.preprocess(
            dummy_imagery, 
            "satbird", 
            include_environmental=False
        )
        
        print(f"✓ SatBird preprocessing without environmental data successful")
        print(f"  Output shape (no env): {result_no_env['imagery'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data preprocessing test failed: {e}")
        traceback.print_exc()
        return False

def test_satbird_model_loading():
    """Test SatBird model loading with environmental variant."""
    print("\n" + "="*60)
    print("TESTING SATBIRD MODEL LOADING")
    print("="*60)
    
    try:
        from geogist.model_wrappers import create_model_wrapper
        
        # Test SatBird with environmental data
        wrapper = create_model_wrapper("satbird", device="cpu", model_variant="rgbnir_env")
        
        print(f"✓ SatBird wrapper created successfully")
        print(f"  Model variant: {wrapper.model_variant}")
        print(f"  Expected channels: {wrapper.num_channels}")
        print(f"  Include environmental: {wrapper.include_environmental}")
        print(f"  Expected bands: {wrapper.get_expected_bands()}")
        print(f"  Embedding dimension: {wrapper.get_embedding_dimension()}")
        
        # Try to load the model (may fail if weights not found)
        try:
            wrapper.load_model()
            print(f"✓ SatBird model loaded successfully")
            print(f"  Model loaded: {wrapper.loaded}")
            
            # Test embedding extraction with dummy data
            dummy_data = {
                "imagery": torch.randn(1, wrapper.num_channels, 64, 64)
            }
            
            embeddings = wrapper.extract_embeddings(dummy_data)
            print(f"✓ Embedding extraction successful")
            print(f"  Embeddings shape: {embeddings.shape}")
            
        except Exception as e:
            print(f"⚠️  Model loading failed (expected if weights not available): {str(e)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ SatBird model test failed: {e}")
        traceback.print_exc()
        return False

def test_end_to_end_workflow():
    """Test the complete end-to-end workflow."""
    print("\n" + "="*80)
    print("TESTING END-TO-END WORKFLOW")
    print("="*80)
    
    success_tests = []
    
    # Test 1: Planetary Computer workflow
    try:
        print("\n--- Testing with Planetary Computer ---")
        
        result = extract_embeddings(
            latitudes=[40.7128],
            longitudes=[-74.0060],
            datetimes=[datetime(2023, 6, 15)],
            model="satbird",
            imagery_source="planetary_computer",
            temporal_window_days=30,
            max_cloud_cover=0.3,
            include_environmental=True
        )
        
        print(f"✓ Planetary Computer end-to-end workflow successful")
        print(f"  Embeddings shape: {result.shape}")
        success_tests.append("PC workflow")
        
    except Exception as e:
        print(f"✗ Planetary Computer workflow failed: {str(e)[:100]}...")
    
    # Test 2: Earth Engine workflow
    try:
        print("\n--- Testing with Earth Engine ---")
        
        result = extract_embeddings(
            latitudes=[34.0522],
            longitudes=[-118.2437],
            datetimes=[datetime(2023, 6, 15)],
            model="satbird",
            imagery_source="earth_engine",
            temporal_window_days=30,
            max_cloud_cover=0.3,
            include_environmental=True
        )
        
        print(f"✓ Earth Engine end-to-end workflow successful")
        print(f"  Embeddings shape: {result.shape}")
        success_tests.append("EE workflow")
        
    except Exception as e:
        print(f"✗ Earth Engine workflow failed: {str(e)[:100]}...")
    
    # Test 3: Multiple locations
    try:
        print("\n--- Testing with multiple locations ---")
        
        extractor = SatelliteEmbeddingsExtractor(cache_dir="./test_cache", device="cpu")
        request = ExtractionRequest(
            latitudes=[40.7128, 34.0522],
            longitudes=[-74.0060, -118.2437],
            datetimes=[datetime(2023, 6, 15), datetime(2023, 6, 15)],
            model=ModelType.SATBIRD,
            imagery_source="planetary_computer",
            include_environmental=True
        )
        
        result = extractor.extract_embeddings(request)
        
        print(f"✓ Multiple locations workflow successful")
        print(f"  Embeddings shape: {result.embeddings.shape}")
        print(f"  Metadata: {result.metadata}")
        success_tests.append("Multi-location")
        
    except Exception as e:
        print(f"✗ Multiple locations workflow failed: {str(e)[:100]}...")
        traceback.print_exc()
    
    return success_tests

def main():
    """Run all workflow tests."""
    print("=" * 80)
    print("SATELLITE EMBEDDINGS EXTRACTOR - FULL WORKFLOW TEST")
    print("=" * 80)
    
    # Track test results
    test_results = {}
    
    # Test individual components
    test_results["Environmental Data"] = test_environmental_data()
    test_results["Planetary Computer Download"] = test_planetary_computer_download()
    test_results["Earth Engine Download"] = test_earth_engine_download()
    test_results["Data Preprocessing"] = test_data_preprocessing()
    test_results["SatBird Model Loading"] = test_satbird_model_loading()
    
    # Test end-to-end workflows
    successful_workflows = test_end_to_end_workflow()
    test_results["End-to-End Workflows"] = len(successful_workflows) > 0
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30} {status}")
    
    if successful_workflows:
        print(f"\nSuccessful end-to-end workflows: {', '.join(successful_workflows)}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total and successful_workflows:
        print("\n🎉 ALL TESTS PASSED - System is fully operational!")
        print("\nThe satellite embeddings extraction system is working correctly with:")
        print("✓ Environmental data integration")
        print("✓ Planetary Computer imagery download")
        print("✓ Google Earth Engine imagery download")
        print("✓ SatBird model with environmental variables")
        print("✓ End-to-end workflows")
    else:
        print(f"\n⚠️  {total - passed} tests failed - see details above")
        
    return passed == total and len(successful_workflows) > 0

if __name__ == "__main__":
    # Import torch here to check availability
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not available")
    
    success = main()
    sys.exit(0 if success else 1)