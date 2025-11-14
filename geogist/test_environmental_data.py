"""
Comprehensive unit tests for environmental_data.py module.

Tests the EnvironmentalDataDownloader class and its functionality for
downloading and processing WorldClim bioclimatic data and SoilGrids
pedological data.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import requests
import zipfile
import json

from geogist.environmental_data import EnvironmentalDataDownloader, download_environmental_data


class TestEnvironmentalDataDownloader:
    """Test suite for EnvironmentalDataDownloader class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = EnvironmentalDataDownloader(cache_dir=self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test EnvironmentalDataDownloader initialization."""
        assert self.downloader.cache_dir == Path(self.temp_dir)
        assert self.downloader.cache_dir.exists()
        
        # Test bioclim variables
        assert len(self.downloader.bioclim_vars) == 19
        assert 'bio_1' in self.downloader.bioclim_vars
        assert 'bio_19' in self.downloader.bioclim_vars
        
        # Test soil variables
        assert len(self.downloader.soil_vars) == 8
        assert 'bdod' in self.downloader.soil_vars
        assert 'sand' in self.downloader.soil_vars
    
    @patch('geogist.environmental_data.requests.get')
    def test_download_bioclim_data_success(self, mock_get):
        """Test successful WorldClim bioclimatic data download."""
        # Create mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content.return_value = [b'fake_zip_data'] * 10
        mock_get.return_value = mock_response
        
        # Mock zipfile extraction
        with patch('geogist.environmental_data.zipfile.ZipFile') as mock_zip:
            mock_zip_instance = MagicMock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance
            
            result = self.downloader.download_bioclim_data()
            
            assert result == True
            mock_get.assert_called_once()
            mock_zip_instance.extractall.assert_called_once()
    
    @patch('geogist.environmental_data.requests.get')
    def test_download_bioclim_data_already_exists(self, mock_get):
        """Test bioclim download when file already exists."""
        # Create fake existing zip file
        bioclim_dir = self.downloader.cache_dir / "bioclim"
        bioclim_dir.mkdir(exist_ok=True)
        zip_path = bioclim_dir / "wc2.1_30s_bio.zip"
        zip_path.write_text("fake zip content")
        
        result = self.downloader.download_bioclim_data()
        
        assert result == True
        mock_get.assert_not_called()
    
    @patch('geogist.environmental_data.requests.get')
    def test_download_bioclim_data_failure(self, mock_get):
        """Test bioclim download failure handling."""
        mock_get.side_effect = requests.RequestException("Network error")
        
        result = self.downloader.download_bioclim_data()
        
        assert result == False
    
    def test_download_soil_data(self):
        """Test soil data directory preparation."""
        result = self.downloader.download_soil_data()
        
        assert result == True
        soil_dir = self.downloader.cache_dir / "soil"
        assert soil_dir.exists()
    
    def test_get_environmental_stats(self):
        """Test environmental data statistics retrieval."""
        stats = self.downloader.get_environmental_stats()
        
        # Check structure
        assert 'bioclim_means' in stats
        assert 'bioclim_stds' in stats
        assert 'soil_means' in stats
        assert 'soil_stds' in stats
        
        # Check dimensions
        assert len(stats['bioclim_means']) == 19
        assert len(stats['bioclim_stds']) == 19
        assert len(stats['soil_means']) == 8
        assert len(stats['soil_stds']) == 8
        
        # Check data types
        assert stats['bioclim_means'].dtype == np.float32
        assert stats['bioclim_stds'].dtype == np.float32
        assert stats['soil_means'].dtype == np.float32
        assert stats['soil_stds'].dtype == np.float32


class TestBioclimExtraction:
    """Test suite for bioclimatic data extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = EnvironmentalDataDownloader(cache_dir=self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extract_bioclim_no_data(self):
        """Test bioclim extraction when no data files exist."""
        lat, lon, buffer_size = 40.7128, -74.0060, 10
        
        result = self.downloader._extract_bioclim(lat, lon, buffer_size)
        
        assert result.shape == (19, buffer_size, buffer_size)
        assert result.dtype == np.float32
        assert np.all(np.isnan(result))  # Should be all NaN when files don't exist
    
    @patch('geogist.environmental_data.rasterio.open')
    @patch('geogist.environmental_data.zipfile.ZipFile')
    def test_extract_bioclim_with_mock_data(self, mock_zip, mock_rasterio):
        """Test bioclim extraction with mocked rasterio data."""
        lat, lon, buffer_size = 40.7128, -74.0060, 10
        
        # Create fake zip file
        bioclim_dir = self.downloader.cache_dir / "bioclim"
        bioclim_dir.mkdir(exist_ok=True)
        zip_path = bioclim_dir / "wc2.1_30s_bio.zip"
        zip_path.write_text("fake")
        
        # Mock zipfile extraction
        mock_zip_instance = MagicMock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        
        # Mock rasterio reading
        mock_src = MagicMock()
        mock_src.transform = [0.008333, 0, -180, 0, -0.008333, 90]
        mock_src.crs = 'EPSG:4326'
        mock_src.nodata = -9999
        mock_src.read.return_value = np.random.rand(buffer_size, buffer_size) * 300
        mock_rasterio.return_value.__enter__.return_value = mock_src
        
        # Mock rasterio.windows.from_bounds
        with patch('geogist.environmental_data.rasterio.windows.from_bounds') as mock_window:
            mock_window.return_value = MagicMock()
            
            result = self.downloader._extract_bioclim(lat, lon, buffer_size)
            
            assert result.shape == (19, buffer_size, buffer_size)
            assert result.dtype == np.float32
            assert not np.all(np.isnan(result))  # Should have some valid data
    
    def test_extract_bioclim_coordinates_validation(self):
        """Test bioclim extraction with different coordinate inputs."""
        buffer_size = 5
        
        # Test various locations
        test_coords = [
            (0, 0),           # Equator
            (90, 0),          # North pole
            (-90, 0),         # South pole  
            (45, 180),        # Date line
            (45, -180),       # Date line
        ]
        
        for lat, lon in test_coords:
            result = self.downloader._extract_bioclim(lat, lon, buffer_size)
            assert result.shape == (19, buffer_size, buffer_size)
            assert result.dtype == np.float32


class TestSoilExtraction:
    """Test suite for soil data extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = EnvironmentalDataDownloader(cache_dir=self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('geogist.environmental_data.requests.get')
    def test_fetch_soilgrids_point_success(self, mock_get):
        """Test successful SoilGrids API call for single point."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'properties': {
                'bdod': {
                    'layers': [{
                        'depths': [{
                            'values': {'mean': 1450}
                        }]
                    }]
                }
            }
        }
        mock_get.return_value = mock_response
        
        result = self.downloader._fetch_soilgrids_point(40.7, -74.0, 'bdod')
        
        assert result == 1450.0
        mock_get.assert_called_once()
    
    @patch('geogist.environmental_data.requests.get')
    def test_fetch_soilgrids_point_failure(self, mock_get):
        """Test SoilGrids API call failure handling."""
        mock_get.side_effect = requests.RequestException("API error")
        
        result = self.downloader._fetch_soilgrids_point(40.7, -74.0, 'bdod')
        
        assert np.isnan(result)
    
    @patch('geogist.environmental_data.requests.get')
    def test_fetch_soilgrids_point_empty_response(self, mock_get):
        """Test SoilGrids API call with empty response."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {'properties': {}}
        mock_get.return_value = mock_response
        
        result = self.downloader._fetch_soilgrids_point(40.7, -74.0, 'bdod')
        
        assert np.isnan(result)
    
    def test_extract_soil_cached_data(self):
        """Test soil extraction using cached data."""
        lat, lon, buffer_size = 40.7128, -74.0060, 5
        
        # Create fake cached data
        soil_dir = self.downloader.cache_dir / "soil"
        soil_dir.mkdir(exist_ok=True)
        cache_file = soil_dir / f"soil_{lat:.4f}_{lon:.4f}_{buffer_size}.npy"
        
        fake_data = np.random.rand(8, buffer_size, buffer_size).astype(np.float32)
        np.save(cache_file, fake_data)
        
        result = self.downloader._extract_soil(lat, lon, buffer_size)
        
        assert result.shape == (8, buffer_size, buffer_size)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, fake_data)
    
    @patch('geogist.environmental_data.EnvironmentalDataDownloader._fetch_soilgrids_point')
    def test_extract_soil_api_calls(self, mock_fetch):
        """Test soil extraction with mocked API calls."""
        lat, lon, buffer_size = 40.7, -74.0, 3
        
        # Mock API responses
        mock_fetch.return_value = 100.0
        
        # Mock scipy if available
        try:
            import scipy.interpolate
            with patch('geogist.environmental_data.griddata') as mock_griddata:
                mock_griddata.return_value = np.full(buffer_size * buffer_size, 100.0)
                
                result = self.downloader._extract_soil(lat, lon, buffer_size)
                
                assert result.shape == (8, buffer_size, buffer_size)
                assert result.dtype == np.float32
        except ImportError:
            # Test without scipy
            result = self.downloader._extract_soil(lat, lon, buffer_size)
            
            assert result.shape == (8, buffer_size, buffer_size)
            assert result.dtype == np.float32


class TestIntegration:
    """Integration tests for environmental data extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = EnvironmentalDataDownloader(cache_dir=self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extract_environmental_data_structure(self):
        """Test environmental data extraction returns correct structure."""
        latitudes = [40.7128, 34.0522]
        longitudes = [-74.0060, -118.2437]
        buffer_size = 5
        
        results = self.downloader.extract_environmental_data(
            latitudes, longitudes, buffer_size
        )
        
        assert len(results) == 2
        
        for result in results:
            assert 'bioclim' in result
            assert 'soil' in result
            assert 'coordinates' in result
            
            assert result['bioclim'].shape == (19, buffer_size, buffer_size)
            assert result['soil'].shape == (8, buffer_size, buffer_size)
            
            assert result['bioclim'].dtype == np.float32
            assert result['soil'].dtype == np.float32
    
    def test_extract_environmental_data_coordinates_mismatch(self):
        """Test environmental data extraction with mismatched coordinate arrays."""
        latitudes = [40.7128, 34.0522]
        longitudes = [-74.0060]  # Missing one longitude
        
        # Should handle gracefully by stopping at shortest array
        results = self.downloader.extract_environmental_data(latitudes, longitudes)
        
        assert len(results) == 1  # Only one complete coordinate pair
    
    def test_factory_function(self):
        """Test the factory function for creating downloader."""
        downloader = download_environmental_data(self.temp_dir)
        
        assert isinstance(downloader, EnvironmentalDataDownloader)
        assert downloader.cache_dir == Path(self.temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])