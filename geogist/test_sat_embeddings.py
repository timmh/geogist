"""
Comprehensive Unit Tests for Satellite Embeddings Extractor

This module contains unit tests for all components of the satellite embeddings extraction system.
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from geogist.sat_embeddings_extractor import (
    SatelliteEmbeddingsExtractor, 
    ExtractionRequest, 
    ModelType,
    extract_embeddings
)
from geogist.imagery_download import (
    PlanetaryComputerDownloader,
    EarthEngineDownloader,
    SentinelCloudlessDownloader,
    create_downloader
)
from geogist.model_wrappers import (
    SatBirdWrapper,
    GalileoWrapper, 
    PrithviWrapper,
    TesseraWrapper,
    AlphaEarthWrapper,
    TaxaBindWrapper,
    DinoV2Wrapper,
    DinoV3Wrapper,
    create_model_wrapper,
    pool_center_feature_map,
)
from geogist.data_preprocessing import (
    DataPreprocessor,
    TemporalAggregator,
    create_preprocessor
)


class TestSatelliteEmbeddingsExtractor:
    """Test the main SatelliteEmbeddingsExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = SatelliteEmbeddingsExtractor(
            device="cpu"
        )
        
        # Test data
        self.test_lats = [40.7128, 34.0522]
        self.test_lons = [-74.0060, -118.2437]
        self.test_times = [datetime(2023, 6, 15), datetime(2023, 6, 15)]
    
    def test_init(self):
        """Test SatelliteEmbeddingsExtractor initialization."""
        assert self.extractor.device == torch.device("cpu")
        assert self.extractor._models == {}
    
    def test_device_setup_auto(self):
        """Test automatic device setup."""
        extractor = SatelliteEmbeddingsExtractor(device="auto")
        assert isinstance(extractor.device, torch.device)
    
    def test_device_setup_manual(self):
        """Test manual device setup."""
        extractor = SatelliteEmbeddingsExtractor(device="cpu")
        assert extractor.device == torch.device("cpu")
    
    def test_validate_request_valid(self):
        """Test request validation with valid inputs."""
        request = ExtractionRequest(
            latitudes=self.test_lats,
            longitudes=self.test_lons,
            datetimes=self.test_times,
            model=ModelType.SATBIRD
        )
        # Should not raise any exception
        self.extractor._validate_request(request)
    
    def test_validate_request_mismatched_lengths(self):
        """Test request validation with mismatched input lengths."""
        request = ExtractionRequest(
            latitudes=[40.7128],
            longitudes=[-74.0060, -118.2437],  # Different length
            datetimes=self.test_times,
            model=ModelType.SATBIRD
        )
        with pytest.raises(ValueError, match="must have the same length"):
            self.extractor._validate_request(request)
    
    def test_validate_request_empty_inputs(self):
        """Test request validation with empty inputs."""
        request = ExtractionRequest(
            latitudes=[],
            longitudes=[],
            datetimes=[],
            model=ModelType.SATBIRD
        )
        with pytest.raises(ValueError, match="At least one location"):
            self.extractor._validate_request(request)
    
    def test_validate_request_invalid_latitude(self):
        """Test request validation with invalid latitude."""
        request = ExtractionRequest(
            latitudes=[95.0],  # Invalid latitude
            longitudes=[-74.0060],
            datetimes=[datetime.now()],
            model=ModelType.SATBIRD
        )
        with pytest.raises(ValueError, match="Invalid latitude"):
            self.extractor._validate_request(request)
    
    def test_validate_request_invalid_longitude(self):
        """Test request validation with invalid longitude."""
        request = ExtractionRequest(
            latitudes=[40.7128],
            longitudes=[200.0],  # Invalid longitude
            datetimes=[datetime.now()],
            model=ModelType.SATBIRD
        )
        with pytest.raises(ValueError, match="Invalid longitude"):
            self.extractor._validate_request(request)
    
    def test_validate_request_invalid_imagery_source(self):
        """Test request validation with invalid imagery source."""
        request = ExtractionRequest(
            latitudes=self.test_lats,
            longitudes=self.test_lons,
            datetimes=self.test_times,
            model=ModelType.SATBIRD,
            imagery_source="invalid_source"
        )
        with pytest.raises(ValueError, match="sentinel_cloudless"):
            self.extractor._validate_request(request)
    
    def test_validate_request_forces_cloudless_source(self):
        """Cloudless models should force the Sentinel-2 cloudless source."""
        request = ExtractionRequest(
            latitudes=self.test_lats,
            longitudes=self.test_lons,
            datetimes=self.test_times,
            model=ModelType.TAXABIND,
            imagery_source="planetary_computer",
        )
        with pytest.warns(UserWarning, match="cloudless imagery source"):
            self.extractor._validate_request(request)
        assert request.imagery_source == "sentinel_cloudless"

    def test_validate_request_rejects_cloudless_for_other_models(self):
        """Non cloudless models cannot use the Sentinel cloudless source."""
        request = ExtractionRequest(
            latitudes=self.test_lats,
            longitudes=self.test_lons,
            datetimes=self.test_times,
            model=ModelType.SATBIRD,
            imagery_source="sentinel_cloudless",
        )
        with pytest.raises(ValueError, match="only supported"):
            self.extractor._validate_request(request)

    def test_validate_request_negative_scale(self):
        """Scale must be positive when provided."""
        request = ExtractionRequest(
            latitudes=self.test_lats,
            longitudes=self.test_lons,
            datetimes=self.test_times,
            model=ModelType.SATBIRD,
            scale=-5.0,
        )
        with pytest.raises(ValueError, match="positive"):
            self.extractor._validate_request(request)

    def test_validate_request_infinite_scale(self):
        """Scale must be finite."""
        request = ExtractionRequest(
            latitudes=self.test_lats,
            longitudes=self.test_lons,
            datetimes=self.test_times,
            model=ModelType.SATBIRD,
            scale=float('inf'),
        )
        with pytest.raises(ValueError, match="finite"):
            self.extractor._validate_request(request)
    
    def test_validate_request_tessera_valid(self):
        """Test request validation with TESSERA model."""
        request = ExtractionRequest(
            latitudes=self.test_lats,
            longitudes=self.test_lons,
            datetimes=self.test_times,
            model=ModelType.TESSERA,
            imagery_source="invalid_source"  # Should be ignored for TESSERA
        )
        # Should not raise any exception for TESSERA even with invalid imagery source
        self.extractor._validate_request(request)
    
    def test_download_imagery_tessera_skips_download(self):
        """Test that TESSERA skips imagery download."""
        request = ExtractionRequest(
            latitudes=self.test_lats,
            longitudes=self.test_lons,
            datetimes=self.test_times,
            model=ModelType.TESSERA
        )
        
        imagery_data = self.extractor._download_imagery(request)
        
        assert "coordinates" in imagery_data
        assert "datetimes" in imagery_data
        assert "metadata" in imagery_data
        assert imagery_data["metadata"]["source"] == "tessera"
        assert imagery_data["metadata"]["skip_download"] == True

    def test_download_imagery_alphaearth_skips_download(self):
        """Test that AlphaEarth skips imagery download."""
        request = ExtractionRequest(
            latitudes=self.test_lats,
            longitudes=self.test_lons,
            datetimes=self.test_times,
            model=ModelType.ALPHAEARTH
        )

        imagery_data = self.extractor._download_imagery(request)

        assert imagery_data["metadata"]["source"] == "alphaearth"
        assert imagery_data["metadata"]["skip_download"]


class TestImageryDownloaders:
    """Test imagery download functionality."""
    
    def test_create_downloader_planetary_computer(self):
        """Test creating Planetary Computer downloader."""
        downloader = create_downloader("planetary_computer")
        assert isinstance(downloader, PlanetaryComputerDownloader)
    
    def test_create_downloader_earth_engine(self):
        """Test creating Earth Engine downloader."""
        downloader = create_downloader("earth_engine")
        assert isinstance(downloader, EarthEngineDownloader)

    def test_create_downloader_cloudless(self):
        """Test creating Sentinel cloudless downloader."""
        downloader = create_downloader("sentinel_cloudless")
        assert isinstance(downloader, SentinelCloudlessDownloader)
    
    def test_create_downloader_invalid(self):
        """Test creating downloader with invalid source."""
        with pytest.raises(ValueError, match="Unknown imagery source"):
            create_downloader("invalid_source")
    
    def test_planetary_computer_downloader_init(self):
        """Test Planetary Computer downloader initialization."""
        downloader = PlanetaryComputerDownloader()
        assert "satbird" in downloader.band_configs
        assert "galileo" in downloader.band_configs
        assert "prithvi_v2" in downloader.band_configs
    
    def test_planetary_computer_get_bands_for_model(self):
        """Test getting bands for different models."""
        downloader = PlanetaryComputerDownloader()
        
        satbird_bands = downloader.get_bands_for_model("satbird")
        assert satbird_bands == ["B02", "B03", "B04", "B08"]
        
        galileo_bands = downloader.get_bands_for_model("galileo")
        assert len(galileo_bands) == 10
    
        prithvi_bands = downloader.get_bands_for_model("prithvi_v2")
        assert len(prithvi_bands) == 6
    
    def test_planetary_computer_get_bands_unknown_model(self):
        """Test getting bands for unknown model."""
        downloader = PlanetaryComputerDownloader()
        with pytest.raises(ValueError, match="Unknown model"):
            downloader.get_bands_for_model("unknown_model")
    
    def test_create_bbox(self):
        """Test bounding box creation."""
        downloader = PlanetaryComputerDownloader()
        bbox = downloader._create_bbox(40.7128, -74.0060, (64, 64))
        
        assert bbox["type"] == "Polygon"
        assert len(bbox["coordinates"][0]) == 5  # Closed polygon
        assert isinstance(bbox["coordinates"][0][0][0], float)  # Longitude
        assert isinstance(bbox["coordinates"][0][0][1], float)  # Latitude

    def test_cloudless_bbox_symmetry(self):
        """Sentinel cloudless bbox should be centered on the point."""
        downloader = SentinelCloudlessDownloader(pixel_size_meters=20.0)
        minx, miny, maxx, maxy = downloader._compute_bbox(0.0, 0.0, 128, 128)
        assert np.isclose(maxx - minx, 128 * 20.0)
        assert np.isclose(maxy - miny, 128 * 20.0)
        assert np.isclose((maxx + minx) / 2.0, 0.0, atol=1e-3)

    def test_cloudless_infers_year_from_datetime(self):
        """Sentinel cloudless downloader should derive layers from requested datetimes."""
        downloader = SentinelCloudlessDownloader()
        fake_patch = np.zeros((3, 4, 4), dtype=np.float32)
        with patch.object(downloader, "_fetch_patch", return_value=fake_patch) as mock_fetch:
            data = downloader.download_for_locations(
                latitudes=[0.0, 0.0],
                longitudes=[0.0, 0.0],
                datetimes=[datetime(2014, 1, 1), datetime(2025, 6, 1)],
                bands=["R", "G", "B"],
                image_size=(4, 4),
            )
        assert mock_fetch.call_count == 2
        layers = [meta["layer"] for meta in data["metadata"]]
        assert layers[0].startswith("s2cloudless-2016")
        assert layers[1].startswith("s2cloudless-2024")


class TestModelWrappers:
    """Test model wrapper functionality."""
    
    def test_create_model_wrapper_satbird(self):
        """Test creating SatBird wrapper."""
        wrapper = create_model_wrapper("satbird")
        assert isinstance(wrapper, SatBirdWrapper)
    
    def test_create_model_wrapper_galileo(self):
        """Test creating Galileo wrapper."""
        wrapper = create_model_wrapper("galileo")
        assert isinstance(wrapper, GalileoWrapper)
    
    def test_create_model_wrapper_prithvi(self):
        """Test creating Prithvi wrapper."""
        wrapper = create_model_wrapper("prithvi_v2")
        assert isinstance(wrapper, PrithviWrapper)
    
    def test_create_model_wrapper_tessera(self):
        """Test creating TESSERA wrapper."""
        wrapper = create_model_wrapper("tessera")
        assert isinstance(wrapper, TesseraWrapper)

    def test_create_model_wrapper_alphaearth(self):
        """Test creating AlphaEarth wrapper."""
        wrapper = create_model_wrapper("alphaearth")
        assert isinstance(wrapper, AlphaEarthWrapper)

    def test_create_model_wrapper_taxabind(self):
        """Test creating TaxaBind wrapper."""
        wrapper = create_model_wrapper("taxabind")
        assert isinstance(wrapper, TaxaBindWrapper)

    def test_create_model_wrapper_dinov2(self):
        """Test creating DINOv2 wrapper."""
        wrapper = create_model_wrapper("dinov2")
        assert isinstance(wrapper, DinoV2Wrapper)

    def test_create_model_wrapper_dinov3(self):
        """Test creating DINOv3 wrapper."""
        wrapper = create_model_wrapper("dinov3")
        assert isinstance(wrapper, DinoV3Wrapper)
    
    def test_create_model_wrapper_invalid(self):
        """Test creating wrapper with invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model_wrapper("invalid_model")
    
    def test_satbird_wrapper_init(self):
        """Test SatBird wrapper initialization."""
        wrapper = SatBirdWrapper(device="cpu")
        assert wrapper.device == torch.device("cpu")
        assert wrapper.model_variant == "rgbnir_env"
        assert not wrapper.loaded
        assert len(wrapper.expected_bands) == 4
    
    def test_satbird_wrapper_get_expected_bands(self):
        """Test SatBird expected bands."""
        wrapper = SatBirdWrapper()
        bands = wrapper.get_expected_bands()
        assert bands == ["B02", "B03", "B04", "B08"]
    
    def test_satbird_wrapper_get_embedding_dimension(self):
        """Test SatBird embedding dimension."""
        wrapper = SatBirdWrapper()
        dim = wrapper.get_embedding_dimension()
        assert dim == 512
    
    def test_galileo_wrapper_init(self):
        """Test Galileo wrapper initialization."""
        wrapper = GalileoWrapper(device="cpu")
        assert wrapper.device == torch.device("cpu")
        assert wrapper.model_size == "base"
        assert not wrapper.loaded
        assert len(wrapper.expected_bands) == 10

    def test_galileo_wrapper_extract_embeddings_batches_inputs(self):
        """Galileo wrapper should return one embedding per preprocessed sample."""
        wrapper = GalileoWrapper(device="cpu")
        wrapper.loaded = True
        batch = 3

        mock_model = Mock()

        def fake_forward(**kwargs):
            assert kwargs["s_t_x"].shape[0] == batch
            return (
                torch.ones(
                    batch,
                    8,
                    8,
                    2,
                    3,
                    4,
                    dtype=torch.float32,
                    device=wrapper.device,
                ),
            )

        mock_model.forward.side_effect = fake_forward
        mock_model.embedding_size = 256
        wrapper.model = mock_model

        masked_output = {
            "s_t_x": torch.zeros(batch, 4, 4, 1, 13),
            "s_t_m": torch.zeros(batch, 4, 4, 1, 7),
            "sp_x": torch.zeros(batch, 4, 4, 16),
            "sp_m": torch.zeros(batch, 4, 4, 3),
            "t_x": torch.zeros(batch, 1, 6),
            "t_m": torch.zeros(batch, 1, 3),
            "st_x": torch.zeros(batch, 18),
            "st_m": torch.zeros(batch, 4),
            "months": torch.zeros(batch, 1, dtype=torch.long),
        }
        data = {"masked_output": masked_output}

        embeddings = wrapper.extract_embeddings(data, scale=200.0)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (batch, 4)
        mock_model.forward.assert_called_once()
    
    def test_prithvi_wrapper_init(self):
        """Test Prithvi wrapper initialization."""
        wrapper = PrithviWrapper(device="cpu")
        assert wrapper.device == torch.device("cpu")
        assert wrapper.model_variant == "terratorch_prithvi_eo_v2_600_tl"
        assert not wrapper.loaded
        assert len(wrapper.expected_bands) == 6
    
    def test_tessera_wrapper_init(self):
        """Test TESSERA wrapper initialization."""
        wrapper = TesseraWrapper(device="cpu")
        assert wrapper.device == torch.device("cpu")
        assert wrapper.year == 2024
        assert not wrapper.loaded
        assert len(wrapper.expected_bands) == 0  # TESSERA doesn't use bands
    
    def test_tessera_wrapper_get_expected_bands(self):
        """Test TESSERA expected bands."""
        wrapper = TesseraWrapper()
        bands = wrapper.get_expected_bands()
        assert bands == []
    
    def test_tessera_wrapper_get_embedding_dimension(self):
        """Test TESSERA embedding dimension."""
        wrapper = TesseraWrapper()
        dim = wrapper.get_embedding_dimension()
        assert dim == 128
    
    def test_tessera_wrapper_get_image_size(self):
        """Test TESSERA expected image size."""
        wrapper = TesseraWrapper()
        size = wrapper.get_expected_image_size()
        assert size == (224, 224)
    
    @patch('geotessera.GeoTessera')
    def test_tessera_wrapper_load_model(self, mock_geotessera, tmp_path):
        """Test TESSERA model loading."""
        mock_client = Mock()
        mock_geotessera.return_value = mock_client
        
        wrapper = TesseraWrapper(device="cpu", cache_dir=tmp_path)
        wrapper.load_model()
        
        assert wrapper.loaded
        assert wrapper.model == mock_client
        mock_geotessera.assert_called_once_with(
            cache_dir=str(tmp_path / "tessera" / "cache"),
            embeddings_dir=str(tmp_path / "tessera" / "embeddings"),
        )
        assert (tmp_path / "tessera" / "cache").exists()
        assert (tmp_path / "tessera" / "embeddings").exists()
    
    @patch('rasterio.transform.rowcol')
    @patch('geotessera.GeoTessera')
    def test_tessera_wrapper_extract_embeddings(self, mock_geotessera, mock_rowcol, tmp_path):
        """Test TESSERA embedding extraction."""
        # Mock the GeoTessera client
        mock_client = Mock()
        mock_embedding = np.random.rand(200, 200, 128)  # Mock spatial embedding
        mock_client.fetch_embedding.return_value = (mock_embedding, None, SimpleNamespace(a=10, e=-10))
        mock_geotessera.return_value = mock_client
        mock_rowcol.return_value = (100, 100)
        
        wrapper = TesseraWrapper(device="cpu", cache_dir=tmp_path)
        wrapper.load_model()
        
        # Test data
        test_data = {
            'latitudes': [40.7128, 34.0522],
            'longitudes': [-74.0060, -118.2437],
            'datetimes': [datetime(2023, 6, 15), datetime(2023, 6, 15)]
        }
        
        embeddings = wrapper.extract_embeddings(test_data)
        
        # Check output shape and type
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 128)  # 2 locations, 128-dim embeddings
        assert mock_client.fetch_embedding.call_count == 2

    @patch('rasterio.transform.rowcol')
    @patch('geotessera.GeoTessera')
    def test_tessera_wrapper_scale_clipping_warns(self, mock_geotessera, mock_rowcol, tmp_path):
        mock_client = Mock()
        mock_embedding = np.arange(16, dtype=float).reshape(4, 4, 1)
        transform = SimpleNamespace(a=10, e=-10)
        mock_client.fetch_embedding.return_value = (mock_embedding, None, transform)
        mock_geotessera.return_value = mock_client
        mock_rowcol.return_value = (2, 2)

        wrapper = TesseraWrapper(device="cpu", cache_dir=tmp_path)
        wrapper.load_model()

        test_data = {
            'latitudes': [40.0],
            'longitudes': [-120.0],
            'datetimes': [datetime(2023, 6, 15)]
        }

        with pytest.warns(UserWarning, match="clipping"):
            embeddings = wrapper.extract_embeddings(test_data, scale=500.0)

        assert embeddings.shape == (1, 1)
        np.testing.assert_allclose(embeddings[0], mock_embedding.mean(axis=(0, 1)))

    def test_alphaearth_wrapper_init(self):
        wrapper = AlphaEarthWrapper(device="cpu")
        assert wrapper.device == torch.device("cpu")
        assert wrapper.embedding_dimension == 64
        assert wrapper.expected_bands[0] == "A00"

    def test_alphaearth_wrapper_load_model(self):
        mock_module = SimpleNamespace(
            Initialize=Mock(),
            ImageCollection=Mock(return_value="collection")
        )
        with patch.dict('sys.modules', {'ee': mock_module}):
            wrapper = AlphaEarthWrapper(device="cpu")
            wrapper.load_model()
        mock_module.Initialize.assert_called_once()
        mock_module.ImageCollection.assert_called_once()
        assert wrapper.loaded
        assert wrapper.collection == "collection"

    def test_alphaearth_wrapper_extract_embeddings(self):
        wrapper = AlphaEarthWrapper(device="cpu")
        wrapper.loaded = True
        wrapper.ee = SimpleNamespace()
        wrapper.collection = object()
        patch_array = np.ones((3, 3, wrapper.embedding_dimension), dtype=np.float32)
        with patch.object(wrapper, "_get_patch_for_location", return_value=patch_array) as mock_get_patch:
            data = {
                'latitudes': [40.0],
                'longitudes': [-120.0],
                'datetimes': [datetime(2023, 6, 15)]
            }
            embeddings = wrapper.extract_embeddings(data, scale=30.0)
        mock_get_patch.assert_called_once()
        assert embeddings.shape == (1, wrapper.embedding_dimension)
        assert np.allclose(embeddings, embeddings[0])

    def test_alphaearth_wrapper_extract_embeddings_handles_missing_patch(self):
        wrapper = AlphaEarthWrapper(device="cpu")
        wrapper.loaded = True
        wrapper.ee = SimpleNamespace()
        wrapper.collection = object()
        with patch.object(wrapper, "_get_patch_for_location", return_value=None):
            data = {
                'latitudes': [40.0],
                'longitudes': [-120.0],
                'datetimes': [datetime(2023, 6, 15)]
            }
            embeddings = wrapper.extract_embeddings(data)
        assert embeddings.shape == (1, wrapper.embedding_dimension)
        assert np.all(embeddings == 0)

    def test_dinov3_wrapper_requires_weights(self):
        wrapper = DinoV3Wrapper(device="cpu")
        with pytest.raises(ValueError, match="weights_path"):
            wrapper.load_model()

    def test_taxabind_wrapper_extracts_with_mock(self):
        mock_model = Mock()
        mock_model.encode_image.return_value = torch.zeros(1, 512)
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        fake_open_clip = SimpleNamespace(create_model_and_transforms=Mock(return_value=(mock_model, None, None)))
        wrapper = TaxaBindWrapper(device="cpu")
        with patch.dict('sys.modules', {'open_clip': fake_open_clip}):
            wrapper.load_model()
        data = {"imagery": torch.zeros(1, 3, 224, 224)}
        embeddings = wrapper.extract_embeddings(data)
        assert embeddings.shape == (1, 512)

    @patch('torch.hub.load')
    def test_dinov2_wrapper_uses_torch_hub(self, mock_hub_load):
        mock_model = Mock()
        mock_model.return_value = torch.zeros(1, 768)
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_hub_load.return_value = mock_model
        wrapper = DinoV2Wrapper(device="cpu")
        wrapper.load_model()
        data = {"imagery": torch.zeros(1, 3, 224, 224)}
        wrapper.extract_embeddings(data)
        mock_hub_load.assert_called_once()

    @patch('torch.hub.load')
    def test_dinov3_wrapper_loads_with_weights(self, mock_hub_load):
        mock_model = Mock()
        mock_model.return_value = torch.zeros(1, 768)
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_hub_load.return_value = mock_model
        wrapper = DinoV3Wrapper(device="cpu", weights_path="/tmp/dinov3.pth")
        wrapper.load_model()
        data = {"imagery": torch.zeros(1, 3, 224, 224)}
        wrapper.extract_embeddings(data)
        mock_hub_load.assert_called_once()

    def test_satbird_wrapper_uses_custom_weights_path(self, tmp_path):
        """SatBird should accept a custom checkpoint path via weights_path."""
        checkpoint = tmp_path / "custom_satbird.ckpt"
        checkpoint.write_bytes(b"fake")

        class DummySatBirdModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
                self.fc = torch.nn.Linear(512, 1000)

            def forward(self, x):
                batch = x.shape[0]
                return torch.zeros(batch, 512)

        fake_torchvision = ModuleType("torchvision")
        fake_torchvision.models = SimpleNamespace(
            resnet18=lambda pretrained=False: DummySatBirdModel()
        )

        with patch.dict(sys.modules, {"torchvision": fake_torchvision}):
            with patch(
                "geogist.model_wrappers.torch.load",
                return_value={"state_dict": {}},
            ) as mock_load:
                wrapper = SatBirdWrapper(device="cpu", weights_path=checkpoint)
                wrapper.load_model()

        assert wrapper.loaded
        mock_load.assert_called_once()
        assert Path(mock_load.call_args[0][0]) == checkpoint

    @patch("geogist.single_file_galileo.Encoder")
    def test_galileo_wrapper_uses_custom_weights_path(self, mock_encoder, tmp_path):
        """Galileo should allow overriding the model directory."""
        model_dir = tmp_path / "galileo_custom"
        model_dir.mkdir()
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_encoder.load_from_folder.return_value = mock_model

        wrapper = GalileoWrapper(device="cpu", model_size="nano", weights_path=model_dir)
        wrapper.load_model()

        assert wrapper.loaded
        assert wrapper.model == mock_model
        mock_encoder.load_from_folder.assert_called_once_with(
            model_dir, device=wrapper.device
        )


class TestSpatialPooling:
    """Test spatial pooling helper functionality."""

    def test_pool_center_feature_map_defaults_to_center_pixel(self):
        feature_map = torch.arange(25, dtype=torch.float32).reshape(1, 5, 5, 1)
        pooled = pool_center_feature_map(feature_map, None, meters_per_pixel=10.0)
        assert pooled.shape == (1, 1)
        assert torch.allclose(pooled, feature_map[:, 2:3, 2:3, :].reshape(1, 1))

    def test_pool_center_feature_map_warns_when_clipped(self):
        feature_map = torch.ones((1, 2, 2, 1), dtype=torch.float32)
        with pytest.warns(UserWarning, match="clipping"):
            pooled = pool_center_feature_map(feature_map, 100.0, meters_per_pixel=10.0)
        assert pooled.shape == (1, 1)


class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = create_preprocessor()
        
        # Create dummy imagery data
        self.dummy_imagery = {
            "imagery": [np.random.rand(4, 32, 32).astype(np.float32)],
            "metadata": [{"success": True, "cloud_cover": 5.0}],
            "bands": ["B02", "B03", "B04", "B08"],
            "coordinates": [(40.7128, -74.0060)],
            "datetimes": [datetime.now()]
        }
    
    def test_create_preprocessor(self):
        """Test preprocessor creation."""
        preprocessor = create_preprocessor()
        assert isinstance(preprocessor, DataPreprocessor)
    
    def test_preprocess_satbird(self):
        """Test SatBird preprocessing."""
        result = self.preprocessor.preprocess(self.dummy_imagery, "satbird")
        
        assert "imagery" in result
        assert "metadata" in result
        assert isinstance(result["imagery"], torch.Tensor)
        assert result["imagery"].shape == (1, 31, 64, 64)  # [batch, channels, height, width]
    
    def test_preprocess_prithvi(self):
        """Test Prithvi preprocessing."""
        # Create imagery with more bands for Prithvi
        prithvi_imagery = self.dummy_imagery.copy()
        prithvi_imagery["imagery"] = [np.random.rand(6, 32, 32).astype(np.float32)]
        prithvi_imagery["bands"] = ["B02", "B03", "B04", "B08", "B11", "B12"]
        
        result = self.preprocessor.preprocess(prithvi_imagery, "prithvi_v2")
        
        assert "imagery" in result
        assert isinstance(result["imagery"], torch.Tensor)
        assert result["imagery"].shape == (1, 6, 224, 224)  # Prithvi expects 224x224

    def test_preprocess_galileo_batches_all_locations(self):
        """Galileo preprocessing should retain one entry per requested location."""
        galileo_imagery = {
            "imagery": [
                np.random.rand(10, 32, 32).astype(np.float32),
                np.random.rand(10, 32, 32).astype(np.float32),
            ],
            "metadata": [{"success": True}, {"success": True}],
            "bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
            "coordinates": [(10.0, 10.0), (-10.0, -10.0)],
            "datetimes": [datetime.now(), datetime.now()],
        }

        result = self.preprocessor.preprocess(galileo_imagery, "galileo")

        masked_output = result["masked_output"]
        assert isinstance(masked_output, dict)
        assert masked_output["s_t_x"].shape[0] == len(galileo_imagery["imagery"])
        assert masked_output["months"].shape[0] == len(galileo_imagery["imagery"])
    
    def test_preprocess_tessera(self):
        """Test TESSERA preprocessing."""
        result = self.preprocessor.preprocess(self.dummy_imagery, "tessera")
        
        assert "latitudes" in result
        assert "longitudes" in result
        assert "datetimes" in result
        assert "metadata" in result
        assert result["latitudes"] == [40.7128]
        assert result["longitudes"] == [-74.0060]
    
    def test_preprocess_tessera_empty_coordinates(self):
        """Test TESSERA preprocessing with empty coordinates."""
        empty_imagery = self.dummy_imagery.copy()
        empty_imagery["coordinates"] = []
        
        result = self.preprocessor.preprocess(empty_imagery, "tessera")
        
        assert result["latitudes"] == [0]
        assert result["longitudes"] == [0]

    def test_preprocess_alphaearth(self):
        """Test AlphaEarth preprocessing mirrors TESSERA behavior."""
        result = self.preprocessor.preprocess(self.dummy_imagery, "alphaearth")
        assert result["latitudes"] == [40.7128]
        assert result["longitudes"] == [-74.0060]

    def test_preprocess_taxabind_rgb(self):
        """RGB-only models should return normalized tensors."""
        rgb_imagery = {
            "imagery": [np.random.rand(3, 32, 32).astype(np.float32)],
            "metadata": [{}],
            "bands": ["R", "G", "B"],
            "coordinates": [(0.0, 0.0)],
            "datetimes": [datetime.now()],
        }
        result = self.preprocessor.preprocess(rgb_imagery, "taxabind")
        tensor = result["imagery"]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_dinov_models_share_rgb_path(self):
        """DINOv2 and DINOv3 preprocessing should both resize to 224."""
        rgb_imagery = {
            "imagery": [np.random.rand(3, 48, 48).astype(np.float32)],
            "metadata": [{}],
            "bands": ["R", "G", "B"],
            "coordinates": [(0.0, 0.0)],
            "datetimes": [datetime.now()],
        }
        dinov2 = self.preprocessor.preprocess(rgb_imagery, "dinov2")
        dinov3 = self.preprocessor.preprocess(rgb_imagery, "dinov3")
        assert dinov2["imagery"].shape == (1, 3, 224, 224)
        assert dinov3["imagery"].shape == (1, 3, 224, 224)
    
    def test_preprocess_invalid_model(self):
        """Test preprocessing with invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            self.preprocessor.preprocess(self.dummy_imagery, "invalid_model")
    
    def test_resize_imagery(self):
        """Test imagery resizing."""
        imagery = np.random.rand(4, 32, 32)
        resized = self.preprocessor._resize_imagery(imagery, (64, 64))
        assert resized.shape == (4, 64, 64)
    
    def test_normalize_imagery(self):
        """Test imagery normalization."""
        imagery = np.random.rand(4, 32, 32) * 1000  # Large values
        means = np.array([500.0, 500.0, 500.0, 500.0])
        stds = np.array([100.0, 100.0, 100.0, 100.0])
        
        normalized = self.preprocessor._normalize_imagery(imagery, means, stds)
        
        # Check that normalization was applied
        assert normalized.shape == imagery.shape
        assert np.abs(normalized.mean()) < 10  # Should be roughly centered around 0
    
    def test_reorder_bands(self):
        """Test band reordering."""
        imagery = np.random.rand(4, 32, 32)
        current_bands = ["B08", "B04", "B03", "B02"]  # Reversed order
        target_bands = ["B02", "B03", "B04", "B08"]
        
        reordered = self.preprocessor._reorder_bands(imagery, current_bands, target_bands)
        
        assert reordered.shape == imagery.shape
        # Check that bands were reordered (first band should now be last)
        np.testing.assert_array_equal(reordered[0], imagery[3])
    
    def test_handle_missing_data(self):
        """Test handling of missing/NaN data."""
        missing_imagery = self.dummy_imagery.copy()
        missing_imagery["imagery"] = [np.full((4, 32, 32), np.nan)]
        
        result = self.preprocessor.preprocess(missing_imagery, "satbird")
        
        assert "imagery" in result
        assert not torch.isnan(result["imagery"]).any()  # Should be filled with zeros


class TestTemporalAggregation:
    """Test temporal aggregation functionality."""
    
    def test_temporal_aggregator_init(self):
        """Test temporal aggregator initialization."""
        aggregator = TemporalAggregator(method="median")
        assert aggregator.method == "median"
    
    def test_temporal_aggregator_invalid_method(self):
        """Test temporal aggregator with invalid method."""
        aggregator = TemporalAggregator(method="invalid")
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregator.aggregate({})


class TestConvenienceFunction:
    """Test the convenience function."""
    
    @patch('geogist.sat_embeddings_extractor.get_extractor')
    def test_extract_embeddings_convenience_function(self, mock_get_extractor):
        """Test the convenience function for extracting embeddings."""
        # Mock the extractor and its methods
        mock_extractor = Mock()
        mock_result = Mock()
        mock_result.embeddings = np.random.rand(2, 512)
        mock_extractor.extract_embeddings.return_value = mock_result
        mock_get_extractor.return_value = mock_extractor
        
        # Test the convenience function
        result = extract_embeddings(
            latitudes=[40.7128, 34.0522],
            longitudes=[-74.0060, -118.2437],
            datetimes=[datetime.now(), datetime.now()],
            model="satbird",
            scale=250.0,
        )
        
        # Verify the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 512)
        
        # Verify the extractor was called correctly
        mock_get_extractor.assert_called_once_with(cache_dir=None)
        mock_extractor.extract_embeddings.assert_called_once()
        request_arg = mock_extractor.extract_embeddings.call_args[0][0]
        assert request_arg.scale == 250.0

    @patch('geogist.sat_embeddings_extractor.get_extractor')
    def test_extract_embeddings_custom_cache_dir(self, mock_get_extractor):
        """Ensure cache_dir argument is forwarded to the extractor factory."""
        mock_extractor = Mock()
        mock_result = Mock()
        mock_result.embeddings = np.random.rand(1, 512)
        mock_extractor.extract_embeddings.return_value = mock_result
        mock_get_extractor.return_value = mock_extractor
        custom_cache = Path("/tmp/geogist-test-cache")

        extract_embeddings(
            latitudes=[40.0],
            longitudes=[-120.0],
            datetimes=[datetime.now()],
            model="satbird",
            cache_dir=custom_cache,
        )

        mock_get_extractor.assert_called_once_with(cache_dir=custom_cache)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
