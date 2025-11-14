"""
Model Wrappers for Geospatial Foundation Models

This module provides wrapper classes for SatBird, Galileo, and Prithvi V2 models
to enable consistent embedding extraction.
"""

import math
import warnings
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from datetime import datetime

# Import package path utilities
from .paths import (
    get_package_root,
    get_weights_path,
    get_cache_path,
    ensure_path_exists,
)

ALPHAEARTH_BANDS: List[str] = [f"A{i:02d}" for i in range(64)]
RGB_BANDS: List[str] = ["R", "G", "B"]


class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.model = None
        self.loaded = False
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup the computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def extract_embeddings(self, data: Dict[str, Any], scale: Optional[float] = None) -> np.ndarray:
        """Extract embeddings from preprocessed data."""
        pass
    
    @abstractmethod
    def get_expected_bands(self) -> List[str]:
        """Get the bands expected by this model."""
        pass

    @abstractmethod
    def get_expected_image_size(self) -> Tuple[int, int]:
        """Get the expected input image size for this model."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass


def pool_center_feature_map(
    feature_map: torch.Tensor,
    scale_meters: Optional[float],
    meters_per_pixel: float,
    warn_fn: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """Pool a spatial feature map around its center using the requested scale."""
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [batch, height, width, channels]")
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive")

    if warn_fn is None:
        warn_fn = warnings.warn

    _, height, width, _ = feature_map.shape
    if height == 0 or width == 0:
        raise ValueError("feature_map must have non-zero spatial dimensions")

    max_extent_m = min(height, width) * meters_per_pixel
    requested_m = None if scale_meters is None else float(scale_meters)

    if requested_m is None or requested_m <= 0:
        span = 1
        clipped = False
    else:
        if math.isfinite(requested_m):
            span = max(1, math.ceil(requested_m / meters_per_pixel))
        else:
            span = min(height, width)
        clipped = requested_m > max_extent_m + 1e-6

    span = min(span, min(height, width))
    row_start = max(0, (height - span) // 2)
    col_start = max(0, (width - span) // 2)
    row_end = row_start + span
    col_end = col_start + span

    region = feature_map[:, row_start:row_end, col_start:col_end, :]
    pooled = region.mean(dim=(1, 2))

    if clipped and warn_fn is not None:
        warn_fn(
            f"Requested pooling scale {requested_m:.1f}m exceeds available {max_extent_m:.1f}m; "
            f"clipping to {span * meters_per_pixel:.1f}m."
        )

    return pooled


class SatBirdWrapper(ModelWrapper):
    """Wrapper for SatBird model."""
    
    def __init__(
        self,
        device: str = "auto",
        model_variant: str = "rgbnir_env",
        weights_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__(device)
        self.model_variant = model_variant
        self.weights_path = (
            Path(weights_path).expanduser() if weights_path is not None else None
        )
        
        # Expected bands based on variant
        if model_variant == "rgbnir":
            self.expected_bands = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR
            self.num_channels = 4
        elif model_variant == "rgbnir_env":
            self.expected_bands = ["B02", "B03", "B04", "B08"]  # Will add env later
            self.num_channels = 4 + 19 + 8  # RGBNIR + bioclim + pedological
        else:
            raise ValueError(f"Unknown SatBird variant: {model_variant}")
        
        self.expected_image_size = (64, 64)  # SatBird input size
        
        self.include_environmental = model_variant == "rgbnir_env"
    
    def load_model(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        """
        Load SatBird model from checkpoint.
        
        Args:
            checkpoint_path: Optional override for SatBird checkpoint file
        """
        if checkpoint_path is not None:
            resolved_checkpoint = Path(checkpoint_path)
        elif self.weights_path is not None:
            resolved_checkpoint = Path(self.weights_path)
        else:
            # Use default weights path
            resolved_checkpoint = get_weights_path() / "satbird/satbird_rgbnir_epoch=38-step=26090.ckpt"
        
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"SatBird checkpoint not found: {resolved_checkpoint}")
        
        # Import SatBird modules
        try:
            from torchvision import models
        except ImportError as e:
            raise ImportError(f"Failed to import required modules for SatBird: {e}")
        
        # Create ResNet18 model with appropriate input channels
        self.model = models.resnet18(pretrained=False)
        
        # Modify first layer for multi-channel input
        if self.num_channels != 3:
            weights = self.model.conv1.weight.data.clone()
            self.model.conv1 = nn.Conv2d(
                self.num_channels, 64, 
                kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            # Initialize weights (basic approach - could be improved)
            self.model.conv1.weight.data = self._init_first_layer_weights(self.num_channels, weights)
        
        # Load checkpoint - use weights_only=False for trusted checkpoint with OmegaConf config
        checkpoint = torch.load(resolved_checkpoint, map_location=self.device, weights_only=False)
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        
        # Remove final classification layer
        if 'fc.weight' in state_dict:
            del state_dict['fc.weight']
        if 'fc.bias' in state_dict:
            del state_dict['fc.bias']
        
        # Load weights
        self.model.load_state_dict(state_dict, strict=False)
        
        # Remove final classification layer for feature extraction
        self.model.fc = nn.Identity()
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True
        
        print(f"SatBird model loaded successfully on {self.device}")
    
    def _init_first_layer_weights(self, num_channels: int, original_weights: torch.Tensor) -> torch.Tensor:
        """Initialize first layer weights for multi-channel input."""
        out_channels, _, kernel_h, kernel_w = original_weights.shape
        new_weights = torch.zeros(out_channels, num_channels, kernel_h, kernel_w)
        
        # Copy RGB channels if available
        if num_channels >= 3:
            new_weights[:, :3, :, :] = original_weights[:, :3, :, :]
        
        # Initialize additional channels with small random values
        if num_channels > 3:
            new_weights[:, 3:, :, :] = torch.randn(out_channels, num_channels - 3, kernel_h, kernel_w) * 0.01
        
        return new_weights
    
    def extract_embeddings(self, data: Dict[str, Any], scale: Optional[float] = None) -> np.ndarray:
        """
        Extract embeddings from preprocessed data.

        Args:
            data: Dictionary containing 'imagery' key with preprocessed image tensors
            scale: Optional pooling scale (ignored because SatBird outputs global embeddings)
            
        Returns:
            Array of embeddings with shape [batch_size, embedding_dim]
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        imagery = data['imagery']
        if isinstance(imagery, np.ndarray):
            imagery = torch.from_numpy(imagery).float()
        
        # Ensure tensor is on correct device
        imagery = imagery.to(self.device)
        
        # Add batch dimension if needed
        if len(imagery.shape) == 3:
            imagery = imagery.unsqueeze(0)
        
        with torch.no_grad():
            embeddings = self.model(imagery)
        
        return embeddings.cpu().numpy()
    
    def get_expected_bands(self) -> List[str]:
        """Get the bands expected by SatBird."""
        return self.expected_bands
    
    def get_expected_image_size(self) -> Tuple[int, int]:
        """Get the expected input image size for SatBird."""
        return self.expected_image_size
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by SatBird."""
        return 512  # ResNet18 feature dimension


class GalileoWrapper(ModelWrapper):
    """Wrapper for Galileo model."""
    
    def __init__(
        self,
        device: str = "auto",
        model_size: str = "base",
        weights_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__(device)
        self.model_size = model_size
        self.weights_path = Path(weights_path).expanduser() if weights_path is not None else None
        self.expected_bands = [
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"
        ]  # Full Sentinel-2 bands
        self.expected_image_size = (224, 224)
        self.pixel_resolution_m = 10.0  # Sentinel-2 Level-2A resolution
    
    def load_model(
        self,
        model_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        """
        Load Galileo model.
        
        Args:
            model_path: Optional override path to Galileo model directory
        """
        if model_path is not None:
            resolved_model_path = Path(model_path)
        elif self.weights_path is not None:
            resolved_model_path = self.weights_path
        else:
            resolved_model_path = get_weights_path() / f"galileo/models/{self.model_size}"
            
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Galileo model not found: {resolved_model_path}")
        
        try:
            # Import Galileo modules
            from geogist.single_file_galileo import Encoder
            
            # Load model
            self.model = Encoder.load_from_folder(resolved_model_path, device=self.device)
            self.model.eval()
            self.loaded = True
            
            print(f"Galileo {self.model_size} model loaded successfully on {self.device}")
            
        except ImportError as e:
            raise ImportError(f"Failed to import Galileo modules: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Galileo model: {e}")
    
    def extract_embeddings(self, data: Dict[str, Any], scale: Optional[float] = None) -> np.ndarray:
        """
        Extract embeddings from preprocessed Galileo data.

        Args:
            data: Dictionary containing Galileo input data
            scale: Optional pooling scale in meters for the spatial feature map

        Returns:
            Array of embeddings with shape [batch_size, embedding_dim]
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Extract embeddings using Galileo's forward pass
        try:
            with torch.no_grad():
                galileo_input = self._prepare_batched_input(data.get('masked_output'))

                # Unpack all required arguments for Galileo forward
                output = self.model.forward(
                    s_t_x=galileo_input['s_t_x'],
                    sp_x=galileo_input['sp_x'], 
                    t_x=galileo_input['t_x'],
                    st_x=galileo_input['st_x'],
                    s_t_m=galileo_input['s_t_m'],
                    sp_m=galileo_input['sp_m'],
                    t_m=galileo_input['t_m'],
                    st_m=galileo_input['st_m'],
                    months=galileo_input['months'],
                    patch_size=8  # Use default patch size
                )
                space_time_features = output[0]
                if not isinstance(space_time_features, torch.Tensor):
                    space_time_features = torch.as_tensor(space_time_features, device=self.device)

                # Average over the temporal and channel-group dimensions before spatial pooling
                feature_map = space_time_features.mean(dim=(3, 4))  # [B, H, W, D]
                meters_per_pixel = self._feature_cell_size(feature_map.shape[1])
                embeddings = pool_center_feature_map(feature_map, scale, meters_per_pixel)
            
            return embeddings.cpu().numpy()
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract Galileo embeddings: {e}")

    def _prepare_batched_input(self, masked_output: Any) -> Dict[str, torch.Tensor]:
        """Ensure Galileo inputs include a batch dimension."""
        if masked_output is None:
            raise RuntimeError("No Galileo inputs were provided")

        if isinstance(masked_output, list):
            if not masked_output:
                raise RuntimeError("Empty Galileo input list received")
            reference = masked_output[0]
            return {
                key: torch.stack([sample[key] for sample in masked_output], dim=0)
                for key in reference
            }

        if not isinstance(masked_output, dict):
            raise TypeError("Galileo input must be a dict or list of dicts")

        s_t_x = masked_output.get("s_t_x")
        if not isinstance(s_t_x, torch.Tensor):
            raise TypeError("Galileo input tensors must be torch.Tensor instances")

        if s_t_x.dim() == 4:
            return {key: value.unsqueeze(0) for key, value in masked_output.items()}

        return masked_output

    def _feature_cell_size(self, spatial_size: int) -> float:
        if spatial_size <= 0:
            raise ValueError("Spatial feature map must have positive size")
        image_extent_m = self.expected_image_size[0] * self.pixel_resolution_m
        return image_extent_m / spatial_size
    
    def get_expected_bands(self) -> List[str]:
        """Get the bands expected by Galileo."""
        return self.expected_bands
    
    def get_expected_image_size(self) -> Tuple[int, int]:
        """Get the expected input image size for Galileo."""
        return self.expected_image_size
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by Galileo."""
        return self.model.embedding_size


class PrithviWrapper(ModelWrapper):
    """Wrapper for Prithvi V2 model via TerraTorch."""
    
    def __init__(self, device: str = "auto", model_variant: str = "terratorch_prithvi_eo_v2_600_tl"):
        super().__init__(device)
        self.model_variant = model_variant
        # self.expected_bands = [
        #     "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"
        # ]  # HLS bands
        self.expected_bands = [
            "B02", "B03", "B04", "B08", "B11", "B12"
        ]  # HLS bands
        self.expected_image_size = (224, 224)  # Prithvi input size
        self.pixel_resolution_m = 30.0  # HLS 30m ground sampling distance
    
    def load_model(self, **kwargs) -> None:
        """Load Prithvi V2 model via TerraTorch."""
        try:
            # Import TerraTorch modules
            from terratorch.models.prithvi_model_factory import PrithviModelFactory
            from terratorch.datasets import HLSBands
            
            # Create model factory
            factory = PrithviModelFactory()
            
            # Define bands
            # bands = [HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED, 
            #         HLSBands.RED_EDGE_1, HLSBands.RED_EDGE_2, HLSBands.RED_EDGE_3,
            #         HLSBands.NIR_NARROW, HLSBands.NIR_BROAD, HLSBands.SWIR_1, HLSBands.SWIR_2]
            bands = [HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED, 
                    HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2]
            
            # Build model for feature extraction (no decoder needed)
            from terratorch.registry import BACKBONE_REGISTRY
            self.model = BACKBONE_REGISTRY.build(self.model_variant, pretrained=True)
            
            # Extract just the backbone for feature extraction
            if hasattr(self.model, 'backbone'):
                self.model = self.model.backbone
            
            # Move to device and set to eval mode  
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            
            print(f"Prithvi {self.model_variant} model loaded successfully on {self.device}")
            
        except ImportError as e:
            raise ImportError(
                "Failed to import TerraTorch modules. Install terratorch "
                "(Python >= 3.10) to enable Prithvi models."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Prithvi model: {e}")
    
    def extract_embeddings(self, data: Dict[str, Any], scale: Optional[float] = None) -> np.ndarray:
        """
        Extract embeddings from preprocessed data.
        
        Args:
            data: Dictionary containing 'imagery' key with preprocessed image tensors
            scale: Optional pooling scale in meters for the spatial feature map
        
        Returns:
            Array of embeddings with shape [batch_size, embedding_dim]
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        imagery = data['imagery']
        if isinstance(imagery, np.ndarray):
            imagery = torch.from_numpy(imagery).float()
        
        # Ensure tensor is on correct device
        imagery = imagery.to(self.device)
        
        # Add batch dimension if needed
        if len(imagery.shape) == 3:
            imagery = imagery.unsqueeze(0)
        
        with torch.no_grad():
            model_output = self.model(imagery)
            features = self._coerce_prithvi_output(model_output)
            feature_map = self._reshape_prithvi_feature_map(features)

            if feature_map is None:
                embeddings = self._fallback_prithvi_pool(features)
            else:
                meters_per_pixel = self._feature_cell_size(feature_map.shape[1])
                embeddings = pool_center_feature_map(feature_map, scale, meters_per_pixel)
        
        return embeddings.cpu().numpy()

    def _coerce_prithvi_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, list):
            output = output[-1]
        if hasattr(output, 'output'):
            output = output.output
        elif hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'prediction'):
            output = output.prediction
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            output = torch.as_tensor(output, device=self.device)
        return output

    def _reshape_prithvi_feature_map(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        if features.ndim == 4:  # [B, C, H, W]
            return features.permute(0, 2, 3, 1).contiguous()
        if features.ndim == 3:
            batch, tokens, dim = features.shape
            cls_offset = 0
            side = math.isqrt(tokens)
            if side * side != tokens:
                if tokens <= 1:
                    return None
                potential_side = math.isqrt(tokens - 1)
                if potential_side * potential_side == tokens - 1:
                    cls_offset = 1
                    side = potential_side
                else:
                    return None
            spatial_tokens = features[:, cls_offset:, :]
            return spatial_tokens.reshape(batch, side, side, dim)
        return None

    def _fallback_prithvi_pool(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            return features.unsqueeze(0)
        if features.ndim == 2:
            return features
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 4:
            return features.mean(dim=(2, 3))
        reduce_dims = tuple(range(1, features.ndim))
        return features.mean(dim=reduce_dims)

    def _feature_cell_size(self, spatial_size: int) -> float:
        if spatial_size <= 0:
            raise ValueError("Spatial feature map must have positive size")
        image_extent_m = self.expected_image_size[0] * self.pixel_resolution_m
        return image_extent_m / spatial_size
    
    def get_expected_bands(self) -> List[str]:
        """Get the bands expected by Prithvi."""
        return self.expected_bands
    
    def get_expected_image_size(self) -> Tuple[int, int]:
        """Get the expected input image size for Prithvi."""
        return self.expected_image_size
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by Prithvi."""
        if "300" in self.model_variant:
            return 512  # Prithvi 300M
        elif "600" in self.model_variant:
            return 768  # Prithvi 600M
        else:
            return 512  # Default


class TesseraWrapper(ModelWrapper):
    """Wrapper for TESSERA model via geotessera library."""
    
    def __init__(
        self,
        device: str = "auto",
        year: int = 2024,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(device)
        self.year = year
        self.cache_root = Path(get_cache_path(cache_dir))
        self.expected_bands = []  # TESSERA doesn't use specific bands - it provides pre-computed embeddings
        self.expected_image_size = (224, 224)  # Flexible, can be adjusted
        self.embedding_dimension = 128  # TESSERA provides 128-channel embeddings
        self.tile_size = 2000  # TESSERA tiles are 2000x2000 meters
        self.resolution = 10  # 10m resolution
        
    def load_model(self, **kwargs) -> None:
        """
        Load TESSERA model (geotessera client).
        
        TESSERA doesn't require loading a model - it downloads pre-computed embeddings.
        """
        try:
            from geotessera import GeoTessera
            tessera_root = ensure_path_exists(self.cache_root / "tessera")
            tessera_cache_dir = ensure_path_exists(tessera_root / "cache")
            tessera_embeddings_dir = ensure_path_exists(tessera_root / "embeddings")

            self.model = GeoTessera(
                cache_dir=str(tessera_cache_dir),
                embeddings_dir=str(tessera_embeddings_dir),
            )
            self.loaded = True
            print(f"TESSERA client initialized successfully")
            
        except ImportError as e:
            raise ImportError(f"Failed to import geotessera: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TESSERA client: {e}")
    
    def extract_embeddings(self, data: Dict[str, Any], scale: Optional[float] = None) -> np.ndarray:
        """
        Extract embeddings using TESSERA pre-computed embeddings.
        
        Args:
            data: Dictionary containing location and time information
            scale: Optional pooling scale in meters around the requested coordinate
        
        Returns:
            Array of embeddings with shape [batch_size, embedding_dim]
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Extract coordinates and datetime from data
            latitudes = data['latitudes']
            longitudes = data['longitudes']
            datetimes = data.get('datetimes', [])
            
            # Convert to lists if numpy arrays
            if isinstance(latitudes, np.ndarray):
                latitudes = latitudes.tolist()
            if isinstance(longitudes, np.ndarray):
                longitudes = longitudes.tolist()
            
            embeddings_list = []
            
            for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
                # Determine year from datetime if available, otherwise use default
                year = self.year
                if datetimes and i < len(datetimes):
                    if isinstance(datetimes[i], datetime):
                        year = datetimes[i].year
                    elif isinstance(datetimes[i], str):
                        year = int(datetimes[i][:4])  # Extract year from string


                # TODO: currently, geotessera only has embeddings for 2024
                year = 2024
                
                try:
                    # TESSERA tiles are named by their center coordinates (grid_lon_lat)
                    # Tiles are spaced 0.1° apart, so we need to snap to the nearest grid center
                    # Grid centers are at: ..., -0.05, 0.05, 0.15, 0.25, ...
                    # Each tile covers ±0.05° around its center
                    grid_resolution = 0.1
                    grid_offset = 0.05  # Grid centers are offset by 0.05°

                    # Snap to nearest grid center
                    tile_center_lon = round((float(lon) - grid_offset) / grid_resolution) * grid_resolution + grid_offset
                    tile_center_lat = round((float(lat) - grid_offset) / grid_resolution) * grid_resolution + grid_offset

                    # Fetch embedding for the tile containing this location
                    embedding, crs, transform = self.model.fetch_embedding(
                        lon=tile_center_lon,
                        lat=tile_center_lat,
                        year=year
                    )

                    # Sample embedding at the exact lat/lon coordinates
                    # embedding shape is (H, W, 128)
                    # Convert original lat/lon to pixel coordinates using the transform
                    from rasterio.transform import rowcol
                    row, col = rowcol(transform, float(lon), float(lat))
                    point_embedding = self._pool_tessera_embedding(
                        embedding,
                        row,
                        col,
                        transform,
                        scale,
                    )
                    embeddings_list.append(point_embedding)
                    
                except Exception as e:
                    print(f"Warning: Failed to fetch TESSERA embedding for lat={lat}, lon={lon}, year={year}: {e}")
                    # Use zero embedding as fallback
                    embeddings_list.append(np.zeros(self.embedding_dimension))
            
            # Stack embeddings
            embeddings_array = np.stack(embeddings_list, axis=0)
            return embeddings_array
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract TESSERA embeddings: {e}")
    
    def get_expected_bands(self) -> List[str]:
        """Get the bands expected by TESSERA (none - uses pre-computed embeddings)."""
        return self.expected_bands
    
    def get_expected_image_size(self) -> Tuple[int, int]:
        """Get the expected input image size for TESSERA."""
        return self.expected_image_size

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by TESSERA."""
        return self.embedding_dimension

    def _pool_tessera_embedding(
        self,
        embedding: np.ndarray,
        row: int,
        col: int,
        transform: Any,
        scale: Optional[float],
    ) -> np.ndarray:
        rows, cols, _ = embedding.shape
        row_res, col_res = self._pixel_resolution_from_transform(transform)
        row_span = self._compute_span(scale, row_res, rows)
        col_span = self._compute_span(scale, col_res, cols)

        clipped = False
        if scale is not None:
            max_side = min(rows * row_res, cols * col_res)
            clipped = (row_span == rows or col_span == cols) and (scale > max_side + 1e-6)

        row_start, row_end = self._centered_bounds(row, row_span, rows)
        col_start, col_end = self._centered_bounds(col, col_span, cols)
        region = embedding[row_start:row_end, col_start:col_end, :]
        pooled = region.mean(axis=(0, 1))

        if clipped:
            warnings.warn(
                f"Requested pooling scale {scale:.1f}m exceeds available tile size; "
                f"clipping to {min(row_end - row_start, col_end - col_start) * min(row_res, col_res):.1f}m.",
            )

        return pooled

    def _pixel_resolution_from_transform(self, transform: Any) -> Tuple[float, float]:
        if transform is None:
            return float(self.resolution), float(self.resolution)
        pixel_width = getattr(transform, 'a', None)
        pixel_height = getattr(transform, 'e', None)
        if not pixel_width:
            pixel_width = self.resolution
        if not pixel_height:
            pixel_height = -self.resolution
        return abs(float(pixel_height)), abs(float(pixel_width))

    def _compute_span(self, scale: Optional[float], meters_per_pixel: float, max_size: int) -> int:
        if scale is None or scale <= 0:
            return 1
        span = math.ceil(scale / meters_per_pixel)
        return min(max(span, 1), max_size)

    def _centered_bounds(self, center: int, span: int, max_size: int) -> Tuple[int, int]:
        center = int(center)
        if span >= max_size:
            return 0, max_size
        half = span // 2
        start = center - half
        end = start + span
        if start < 0:
            start = 0
            end = span
        if end > max_size:
            end = max_size
            start = max_size - span
        return start, end


class AlphaEarthWrapper(ModelWrapper):
    """Wrapper for Google AlphaEarth / Satellite Embeddings."""

    def __init__(self,
                 device: str = "auto",
                 collection_id: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
                 default_year: int = 2024):
        super().__init__(device)
        self.collection_id = collection_id
        self.expected_bands = ALPHAEARTH_BANDS
        self.embedding_dimension = len(ALPHAEARTH_BANDS)
        self.expected_image_size = (16384, 16384)
        self.pixel_resolution_m = 10.0
        self.default_year = default_year
        self.max_patch_pixels = 256
        self.ee = None
        self.collection = None

    def load_model(self, **kwargs) -> None:
        """Initialize Earth Engine and reference the AlphaEarth collection."""
        try:
            import ee  # type: ignore
            ee.Initialize()
            self.ee = ee
            self.collection = ee.ImageCollection(self.collection_id)
            self.loaded = True
        except ImportError as exc:
            raise ImportError(f"Failed to import the Earth Engine SDK: {exc}")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Earth Engine for AlphaEarth: {exc}")

    def extract_embeddings(self, data: Dict[str, Any], scale: Optional[float] = None) -> np.ndarray:
        if not self.loaded or self.collection is None or self.ee is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        latitudes = data.get('latitudes', [])
        longitudes = data.get('longitudes', [])
        datetimes = data.get('datetimes', [])

        if isinstance(latitudes, np.ndarray):
            latitudes = latitudes.tolist()
        if isinstance(longitudes, np.ndarray):
            longitudes = longitudes.tolist()

        embeddings_list: List[np.ndarray] = []
        zero_vector = np.zeros(self.embedding_dimension, dtype=np.float32)

        for idx, (lat, lon) in enumerate(zip(latitudes, longitudes)):
            year = self._determine_year(datetimes, idx)
            pixel_span = self._pixel_span_from_scale(scale)
            patch = None
            try:
                patch = self._get_patch_for_location(float(lat), float(lon), year, pixel_span)
            except Exception as exc:
                print(f"Warning: Failed to fetch AlphaEarth embedding for lat={lat}, lon={lon}, year={year}: {exc}")

            if patch is None:
                embeddings_list.append(zero_vector.copy())
                continue

            patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(torch.float32)
            pooled = pool_center_feature_map(patch_tensor, scale, self.pixel_resolution_m)
            embeddings_list.append(pooled.squeeze(0).cpu().numpy())

        if not embeddings_list:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)

        return np.stack(embeddings_list, axis=0)

    def _determine_year(self, datetimes: List[Any], index: int) -> int:
        if datetimes and index < len(datetimes):
            candidate = datetimes[index]
            if isinstance(candidate, datetime):
                return candidate.year
            if isinstance(candidate, str) and len(candidate) >= 4 and candidate[:4].isdigit():
                return int(candidate[:4])
        return self.default_year

    def _pixel_span_from_scale(self, scale: Optional[float]) -> int:
        if scale is None or scale <= 0:
            return 1
        span = max(1, math.ceil(scale / self.pixel_resolution_m))
        return min(span, self.max_patch_pixels)

    def _create_sampling_region(self, lat: float, lon: float, pixel_span: int):
        half_meters = (pixel_span * self.pixel_resolution_m) / 2.0
        lat_deg = half_meters / 111000.0
        cos_lat = math.cos(math.radians(lat))
        lon_scale = max(abs(cos_lat), 1e-6)
        lon_deg = half_meters / (111000.0 * lon_scale)
        return self.ee.Geometry.Rectangle([
            lon - lon_deg,
            lat - lat_deg,
            lon + lon_deg,
            lat + lat_deg
        ])

    def _get_patch_for_location(self, lat: float, lon: float, year: int, pixel_span: int) -> Optional[np.ndarray]:
        point = self.ee.Geometry.Point(lon, lat)
        year = max(2017, min(year, 2024))  # clip year to valid range
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        try:
            collection = (self.collection
                          .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                          .filterBounds(point))
            image = collection.first()
            region = self._create_sampling_region(lat, lon, pixel_span)
            rect = image.sampleRectangle(region=region, defaultValue=None)
            rect_info = rect.getInfo()
        except Exception:
            return None

        band_arrays: List[np.ndarray] = []
        for band in self.expected_bands:
            band_data = rect_info["properties"].get(band)
            if band_data is None:
                return None
            arr = np.array(band_data, dtype=np.float32)
            if arr.size == 0:
                return None
            band_arrays.append(arr)

        try:
            feature_map = np.stack(band_arrays, axis=-1)
        except ValueError:
            return None

        return feature_map

    def get_expected_bands(self) -> List[str]:
        return self.expected_bands

    def get_expected_image_size(self) -> Tuple[int, int]:
        return self.expected_image_size

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension


class TaxaBindWrapper(ModelWrapper):
    """Wrapper for the TaxaBind satellite encoder."""

    def __init__(self, device: str = "auto", model_id: str = "hf-hub:MVRL/taxabind-vit-b-16"):
        super().__init__(device)
        self.model_id = model_id
        self.expected_bands = RGB_BANDS
        self.expected_image_size = (224, 224)
        self.embedding_dimension = 512

    def load_model(self, **kwargs) -> None:
        if self.loaded:
            return
        try:
            import open_clip  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for the TaxaBind wrapper. "
                "Install it with `pip install open-clip-torch`."
            ) from exc
        model, _, _ = open_clip.create_model_and_transforms(self.model_id, device=self.device)
        self.model = model.to(self.device)
        self.model.eval()
        self.loaded = True

    def extract_embeddings(self, data: Dict[str, Any], scale: Optional[float] = None) -> np.ndarray:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        imagery = data["imagery"]
        if isinstance(imagery, np.ndarray):
            imagery = torch.from_numpy(imagery).float()
        imagery = imagery.to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(imagery)
        return embeddings.cpu().numpy()

    def get_expected_bands(self) -> List[str]:
        return self.expected_bands

    def get_expected_image_size(self) -> Tuple[int, int]:
        return self.expected_image_size

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension


class DinoV2Wrapper(ModelWrapper):
    """Wrapper for the DINOv2 ViT-B/14 backbone."""

    def __init__(self, device: str = "auto", variant: str = "dinov2_vitb14"):
        super().__init__(device)
        self.variant = variant
        self.expected_bands = RGB_BANDS
        self.expected_image_size = (224, 224)
        self.embedding_dimension = 768

    def load_model(self, **kwargs) -> None:
        if self.loaded:
            return
        self.model = torch.hub.load("facebookresearch/dinov2", self.variant)
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def extract_embeddings(self, data: Dict[str, Any], scale: Optional[float] = None) -> np.ndarray:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        imagery = data["imagery"]
        if isinstance(imagery, np.ndarray):
            imagery = torch.from_numpy(imagery).float()
        imagery = imagery.to(self.device)
        with torch.no_grad():
            embeddings = self.model(imagery)
        return embeddings.cpu().numpy()

    def get_expected_bands(self) -> List[str]:
        return self.expected_bands

    def get_expected_image_size(self) -> Tuple[int, int]:
        return self.expected_image_size

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension


class DinoV3Wrapper(ModelWrapper):
    """Wrapper for the DINOv3 ViT-B/16 backbone with user-supplied weights."""

    def __init__(
        self,
        device: str = "auto",
        variant: str = "dinov3_vitl16",
        weights_path: Optional[str] = None,
    ):
        super().__init__(device)
        self.variant = variant
        self.weights_path = weights_path
        self.expected_bands = RGB_BANDS
        self.expected_image_size = (224, 224)
        self.embedding_dimension = 768

    def load_model(self, **kwargs) -> None:
        if self.loaded:
            return
        if not self.weights_path:
            raise ValueError(
                "DINOv3 weights_path must be provided via model_kwargs "
                "(e.g., {'weights_path': '/path/to/dinov3_vitb16.pth'})."
            )
        self.model = torch.hub.load(
            "facebookresearch/dinov3",
            self.variant,
            pretrained=True,
            weights=self.weights_path,
        )
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def extract_embeddings(self, data: Dict[str, Any], scale: Optional[float] = None) -> np.ndarray:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        imagery = data["imagery"]
        if isinstance(imagery, np.ndarray):
            imagery = torch.from_numpy(imagery).float()
        imagery = imagery.to(self.device)
        with torch.no_grad():
            embeddings = self.model(imagery)
        return embeddings.cpu().numpy()

    def get_expected_bands(self) -> List[str]:
        return self.expected_bands

    def get_expected_image_size(self) -> Tuple[int, int]:
        return self.expected_image_size

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension


def create_model_wrapper(model_type: str, **kwargs) -> ModelWrapper:
    """
    Factory function to create appropriate model wrapper.
    
    Args:
        model_type: Type of model ("satbird", "galileo", "prithvi_v2", "tessera")
        **kwargs: Additional arguments passed to model wrapper
        
    Returns:
        Initialized model wrapper
    """
    model_type = model_type.lower()
    
    if model_type == "satbird":
        return SatBirdWrapper(**kwargs)
    elif model_type == "galileo":
        return GalileoWrapper(**kwargs)
    elif model_type == "prithvi_v2":
        return PrithviWrapper(**kwargs)
    elif model_type == "tessera":
        return TesseraWrapper(**kwargs)
    elif model_type == "alphaearth":
        return AlphaEarthWrapper(**kwargs)
    elif model_type == "taxabind":
        return TaxaBindWrapper(**kwargs)
    elif model_type == "dinov2":
        return DinoV2Wrapper(**kwargs)
    elif model_type == "dinov3":
        return DinoV3Wrapper(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model wrappers
    print("Testing model wrappers...")
    
    # Test SatBird wrapper creation
    try:
        satbird = create_model_wrapper("satbird")
        print(f"SatBird wrapper created. Expected bands: {satbird.get_expected_bands()}")
        print(f"SatBird embedding dimension: {satbird.get_embedding_dimension()}")
    except Exception as e:
        print(f"SatBird wrapper test failed: {e}")
    
    # Test Galileo wrapper creation
    try:
        galileo = create_model_wrapper("galileo")
        print(f"Galileo wrapper created. Expected bands: {galileo.get_expected_bands()}")
        print(f"Galileo embedding dimension: {galileo.get_embedding_dimension()}")
    except Exception as e:
        print(f"Galileo wrapper test failed: {e}")
    
    # Test Prithvi wrapper creation
    try:
        prithvi = create_model_wrapper("prithvi_v2")
        print(f"Prithvi wrapper created. Expected bands: {prithvi.get_expected_bands()}")
        print(f"Prithvi embedding dimension: {prithvi.get_embedding_dimension()}")
    except Exception as e:
        print(f"Prithvi wrapper test failed: {e}")
    
    # Test TESSERA wrapper creation
    try:
        tessera = create_model_wrapper("tessera")
        print(f"TESSERA wrapper created. Expected bands: {tessera.get_expected_bands()}")
        print(f"TESSERA embedding dimension: {tessera.get_embedding_dimension()}")
    except Exception as e:
        print(f"TESSERA wrapper test failed: {e}")
