# Functions and scripts

## Overview
The core of this project (model and its interface) has been modularized into logical, reusable modules within the `scripts/` folder. The main.py file now acts as a clean orchestrator, importing and calling modularized functions. This document contains explanations for each function in this project.

## Directory Structure

```
scripts/
├── config.py          # Configuration file handling (TOML)
├── constants.py       # All magic numbers and constants
├── utils.py           # Shared utility functions
├── chokepoint.py      # Chokepoint model (YOLOv8) inference
├── fishing.py         # Fishing model (Roboflow) inference
└── dataframe.py       # Result DataFrame construction
```

## Module Breakdown

### `scripts/config.py`
**Purpose**: Centralized configuration management
- `load_config()`: Loads TOML configuration file
- `get_roboflow_api_key()`: Safely retrieves API key with fallback


### `scripts/constants.py`
**Purpose**: Single source of truth for all constants
- Model paths and GitHub URLs
- Confidence thresholds (0.5 for chokepoint, 0.7 for fishing)
- Class names and Roboflow model IDs
- Default image size (640x640)


### `scripts/utils.py`
**Purpose**: Common utility functions used across models
- `extract_datetime_from_image()`: EXIF metadata extraction
- `stretch_image_to_model_size()`: Image preprocessing
- `annotate_image()`: Visualization with bounding boxes


### `scripts/chokepoint.py`
**Purpose**: Chokepoint model (YOLOv8) specific logic
- `load_chokepoint_model()`: Initializes model with GitHub fallback
- `run_chokepoint_inference()`: Runs inference, filters by confidence, counts in/out


### `scripts/fishing.py`
**Purpose**: Fishing model (Roboflow) specific logic
- `get_roboflow_client()`: Lazy-loads Roboflow HTTP client
- `run_fishing_inference()`: Calls Roboflow API, parses predictions


### `scripts/dataframe.py`
**Purpose**: Result formatting and DataFrame construction
- `build_dataframe_row()`: Creates model-specific row format
- `create_dataframe()`: Converts list to pandas DataFrame

## Key Benefits of modularization

1. **Testability**: Each module can be unit tested independently
2. **Maintainability**: Changes to one model don't affect the other
3. **Reusability**: Utils, config, and dataframe modules can be used elsewhere
4. **Scalability**: Easy to add new models - just create `scripts/newmodel.py`
5. **Readability**: The main scripts (main.py) is always clean and easy to follow
6. **Configuration**: All tunable parameters in one place (constants.py)