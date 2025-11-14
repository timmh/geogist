#!/usr/bin/env python3

"""
Command-line interface for the sat-embeddings-helper package.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from .sat_embeddings_extractor import (
    SatelliteEmbeddingsExtractor,
    ExtractionRequest,
    ModelType,
)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract satellite imagery embeddings using geospatial foundation models"
    )
    
    parser.add_argument(
        "--lat", "--latitude", 
        type=float, 
        required=True,
        help="Latitude coordinate"
    )
    
    parser.add_argument(
        "--lon", "--longitude", 
        type=float, 
        required=True,
        help="Longitude coordinate"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "--model",
        choices=["satbird", "galileo", "prithvi_v2", "tessera", "alphaearth", "taxabind", "dinov2", "dinov3"],
        default="satbird",
        help="Foundation model to use"
    )
    
    parser.add_argument(
        "--source",
        choices=["planetary_computer", "earth_engine", "sentinel_cloudless"],
        default="planetary_computer",
        help="Satellite imagery source"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for downloaded data (defaults to a temporary directory)"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run model on"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save embeddings (numpy format)"
    )

    parser.add_argument(
        "--model-weights",
        type=str,
        help="Optional path to model weights (SatBird/Galileo override defaults, required for dinov3)"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse date
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
        
        print(f"Extracting embeddings for ({args.lat}, {args.lon}) on {args.date}")
        print(f"Using model: {args.model}, source: {args.source}")
        
        model_kwargs = {}
        if args.model_weights:
            model_kwargs["weights_path"] = args.model_weights

        extractor = SatelliteEmbeddingsExtractor(
            cache_dir=args.cache_dir,
            device=args.device,
        )

        request = ExtractionRequest(
            latitudes=[args.lat],
            longitudes=[args.lon],
            datetimes=[target_date],
            model=ModelType(args.model),
            imagery_source=args.source,
            model_kwargs=model_kwargs,
        )

        result = extractor.extract_embeddings(request)
        embeddings = result.embeddings
        
        print(f"Successfully extracted embeddings with shape: {embeddings.shape}")
        
        # Save output if specified
        if args.output:
            import numpy as np
            np.save(args.output, embeddings)
            print(f"Embeddings saved to: {args.output}")
        else:
            print("Embedding values:")
            print(embeddings)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
