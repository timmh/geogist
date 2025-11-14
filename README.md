# GeoGist

An opinionated wrapper around various satellite image and geospatial foundation models. GeoGist aims to serve one simple use-case: make it easy to extract embeddings for individual points in space and time.

## Goals

GeoGist has the following goals:
- make it easy to obtain embedding vectors for triplets of latitude, longitude, and time
- abstract away imagery download and model-specific pre-processing
- abstract away model loading and inference

## Non-goals

GeoGist is supposed to do a single thing well. We don't aim to support:
- extracting spatial feature maps
- training or fine-tuning
- benchmarking

## Supported Models

We currently support running inference with the following models:
- [SatBird](https://github.com/RolnickLab/SatBird)
- [Galileo](github.com/nasaharvest/galileo)
- [TaxaBind](https://github.com/mvrl/TaxaBind)
- [Prithvi V2](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) via [TerraTorch](github.com/IBM/terratorch)
- [AlphaEarth / Google Satellite Embedding V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
- [TESSERA](https://github.com/ucam-eo/tessera) via [GeoTessera](https://github.com/ucam-eo/geotessera)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [DINOv3](https://github.com/facebookresearch/dinov3)

## Installation

1. **Install the package:**
```bash
pip install git+https://github.com/timmh/geogist.git
```

2. **Set up satellite imagery access:**

   **For Microsoft Planetary Computer:**
   ```python
   import planetary_computer
   planetary_computer.settings.set_subscription_key("your-api-key")
   ```

   **For Google Earth Engine:**
   ```bash
   earthengine authenticate
   ```

3. **Download model weights:**
   - **SatBird**: After obtaining the original weights, override with `model_kwargs={'weights_path': '/path/to/custom_satbird.ckpt'}` or `--model-weights` in the CLI
   - **Galileo**: Download from [HuggingFace](https://huggingface.co/nasaharvest/galileo)
     (set `weights_path` to the folder containing the encoder checkpoints when needed)
   - **Prithvi V2**: Available through [TerraTorch](https://huggingface.co/ibm-nasa-geospatial)
   - **TaxaBind**: Pulled automatically from Hugging Face (`MVRL/taxabind-vit-b-16`) via `open_clip`
   - **DINOv2**: Pulled automatically through `torch.hub`
   - **DINOv3**: Request access at [ai.meta.com/dinov3](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/), download `dinov3_vitb16` weights locally, and pass the path through `model_kwargs`

## Quick Start

### Basic Usage

```python
from datetime import datetime
from geogist import extract_embeddings

# Define locations and times
latitudes = [40.7128, 34.0522]   # New York, Los Angeles
longitudes = [-74.0060, -118.2437]
datetimes = [datetime(2023, 6, 15), datetime(2023, 6, 15)]

# Extract embeddings using SatBird
embeddings = extract_embeddings(
    latitudes=latitudes,
    longitudes=longitudes,
    datetimes=datetimes,
    model="satbird"
)
```

## Band Statistics Overview

The `compute_band_statistics.py` helper samples roughly 100 random CONUS locations, downloads Sentinel-2 imagery from the configured source (Planetary Computer or Earth Engine) with user-tunable limits on `temporal_window_days`, `max_cloud_cover`, and randomness seed, and computes per-band mean and standard deviation statistics for SatBird, Galileo, and Prithvi inputs. Results are saved to `band_statistics.json` alongside metadata such as the imagery source, sample counts, and the parameter values used. The statistics feed directly into `DataPreprocessor`, which automatically loads them for normalization while falling back to baked-in defaults if the JSON file has not been generated yet.

## Disclaimer
This is experimental code and we do not provide any guarantees that the inference results perfectly match the ones of the upstream processing pipelines. Use at your own risk.

## Acknowledgements & Citations

This library is under the hood merely calling different models. Please consider citing the papers corresponding to models you are using:

```bibtex
# SatBird
@article{teng2023satbird,
  title={Satbird: a dataset for bird species distribution modeling using remote sensing and citizen science data},
  author={Teng, M{\'e}lisande and Elmustafa, Amna and Akera, Benjamin and Bengio, Yoshua and Radi, Hager and Larochelle, Hugo and Rolnick, David},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={75925--75950},
  year={2023}
}

# DINOv2
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}

# Galileo  
@article{tseng2025galileo,
  title={Galileo: Learning global and local features in pretrained remote sensing models},
  author={Tseng, Gabriel and Fuller, Anthony and Reil, Marlena and Herzog, Henry and Beukema, Patrick and Bastani, Favyen and Green, James R and Shelhamer, Evan and Kerner, Hannah and Rolnick, David},
  journal={arXiv e-prints},
  pages={arXiv--2502},
  year={2025}
}

# Prithvi V2
@article{szwarcman2024prithvi,
  title={Prithvi-eo-2.0: A versatile multi-temporal foundation model for earth observation applications},
  author={Szwarcman, Daniela and Roy, Sujit and Fraccaro, Paolo and G{\'\i}slason, {\TH}orsteinn El{\'\i} and Blumenstiel, Benedikt and Ghosal, Rinki and de Oliveira, Pedro Henrique and Almeida, Joao Lucas de Sousa and Sedona, Rocco and Kang, Yanghui and others},
  journal={arXiv preprint arXiv:2412.02732},
  year={2024}
}


# TerraTorch (Prithvi)
@article{gomes2025terratorch,
  title={TerraTorch: The Geospatial Foundation Models Toolkit},
  author={Gomes, Carlos and Blumenstiel, Benedikt and Almeida, Joao Lucas de Sousa and de Oliveira, Pedro Henrique and Fraccaro, Paolo and Escofet, Francesc Marti and Szwarcman, Daniela and Simumba, Naomi and Kienzler, Romeo and Zadrozny, Bianca},
  journal={arXiv preprint arXiv:2503.20563},
  year={2025}
}

# AlphaEarth
@article{brown2025alphaearth,
  title={Alphaearth foundations: An embedding field model for accurate and efficient global mapping from sparse label data},
  author={Brown, Christopher F and Kazmierski, Michal R and Pasquarella, Valerie J and Rucklidge, William J and Samsikova, Masha and Zhang, Chenhui and Shelhamer, Evan and Lahera, Estefania and Wiles, Olivia and Ilyushchenko, Simon and others},
  journal={arXiv preprint arXiv:2507.22291},
  year={2025}
}

# DINOv3
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}


# TaxaBind
@inproceedings{sastry2025taxabind,
    title={TaxaBind: A Unified Embedding Space for Ecological Applications},
    author={Sastry, Srikumar and Khanal, Subash and Dhakal, Aayush and Ahmad, Adeel and Jacobs, Nathan},
    booktitle={Winter Conference on Applications of Computer Vision},
    year={2025},
    organization={IEEE/CVF}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
