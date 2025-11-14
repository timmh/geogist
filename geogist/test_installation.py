#!/usr/bin/env python3

"""
Test script to verify the package installation works correctly.
"""

import sys
import os
import tempfile
from pathlib import Path

def test_basic_import():
    """Test basic package import."""
    print("Testing basic import...")
    
    try:
        import sat_embeddings_helper
        print(f"✓ Package imported successfully. Version: {sat_embeddings_helper.__version__}")
        return True
    except Exception as e:
        print(f"✗ Package import failed: {e}")
        return False

def test_main_components():
    """Test importing main components."""
    print("\nTesting main component imports...")
    
    components = [
        "SatelliteEmbeddingsExtractor",
        "ExtractionRequest", 
        "EmbeddingResult",
        "ModelType",
        "extract_embeddings",
        "ImageryDownloader",
        "ModelWrapper",
        "DataPreprocessor",
    ]
    
    success = True
    for component in components:
        try:
            exec(f"from sat_embeddings_helper import {component}")
            print(f"✓ {component}")
        except Exception as e:
            print(f"✗ {component}: {e}")
            success = False
    
    return success

def test_paths():
    """Test package path utilities."""
    print("\nTesting package paths...")
    
    try:
        from sat_embeddings_helper.paths import (
            get_package_root,
            get_weights_path,
            get_cache_path,
        )
        
        root = get_package_root()
        print(f"✓ Package root: {root}")
        
        weights = get_weights_path()
        print(f"✓ Weights path: {weights}")
        
        cache = get_cache_path("test_cache")
        print(f"✓ Cache path: {cache}")
        
        return True
    except Exception as e:
        print(f"✗ Path utilities failed: {e}")
        return False

def test_extractor_creation():
    """Test creating a SatelliteEmbeddingsExtractor."""
    print("\nTesting extractor creation...")
    
    try:
        from sat_embeddings_helper import SatelliteEmbeddingsExtractor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = SatelliteEmbeddingsExtractor(
                cache_dir=temp_dir,
                device="cpu"
            )
            print(f"✓ Extractor created successfully")
            print(f"  Cache dir: {extractor.cache_dir}")
            print(f"  Device: {extractor.device}")
            
        return True
    except Exception as e:
        print(f"✗ Extractor creation failed: {e}")
        return False

def test_cli_import():
    """Test CLI module import."""
    print("\nTesting CLI import...")
    
    try:
        from sat_embeddings_helper.cli import main
        print("✓ CLI module imported successfully")
        return True
    except Exception as e:
        print(f"✗ CLI import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING SATELLITE EMBEDDINGS HELPER INSTALLATION")
    print("=" * 60)
    
    tests = [
        test_basic_import,
        test_main_components,
        test_paths, 
        test_extractor_creation,
        test_cli_import,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Package installation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
