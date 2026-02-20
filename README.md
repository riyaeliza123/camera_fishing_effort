# Camera Fishing Effort

# IN PROGRESS

FastAPI and Jinja2 web application for uploading fishing camera images, extracting metadata, and preparing data for analysis.

## Features

- 🎯 **Dual Model Support**:
  - **Chokepoint Model**: Detects boats entering/exiting (In/Out detection)
  - **Fishing Model**: Detects boats using Roboflow fishing-access-points/2
- 📸 Image upload with EXIF metadata extraction
- 🤖 YOLOv8-based object detection
- 📊 CSV export of results
- 🚀 Production deployment on Fly.io

## Project Structure

```
camera_fishing_effort/
├── main.py                 # FastAPI application entry point
├── scripts/                # Modularized functionality
│   ├── config.py          # Configuration management
│   ├── constants.py       # Application constants
│   ├── utils.py           # Common utility functions
│   ├── chokepoint.py      # Chokepoint model inference
│   ├── fishing.py         # Fishing model inference (Roboflow)
│   └── dataframe.py       # DataFrame construction
├── static/                # CSS and static assets
├── templates/             # HTML templates
├── config.toml            # Configuration file (gitignored)
├── config.toml.example    # Configuration template
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── fly.toml              # Fly.io configuration
└── README.md             # This file
```

## Setup Instructions

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   ```bash
   # Copy the example file
   cp config.toml.example config.toml
   
   # Edit config.toml and add your Roboflow API key
   # config.toml should look like:
   # [api]
   # roboflow_api_key = "your-api-key-here"
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```
   Visit `http://127.0.0.1:8000` in your browser.

### Production Deployment (Fly.io)

1. **Set the Roboflow API key secret**:
   ```bash
   fly secrets set ROBOFLOW_API_KEY="your-roboflow-api-key"
   ```

2. **Deploy**:
   ```bash
   fly deploy
   ```

## Configuration

### config.toml

Create a `config.toml` file based on `config.toml.example`:

```toml
[api]
roboflow_api_key = "your-roboflow-api-key"
```

### Environment Variables

For Fly.io production:
- `ROBOFLOW_API_KEY`: Your Roboflow API key for the fishing model (required for fishing model)

## Script Modules

### `scripts/config.py`
- `load_config()`: Loads TOML configuration
- `get_roboflow_api_key()`: Retrieves API key from config

### `scripts/constants.py`
- Model paths, URLs, and confidence thresholds
- Class names and Roboflow model IDs
- Default image processing parameters

### `scripts/utils.py`
- `extract_datetime_from_image()`: EXIF metadata extraction
- `stretch_image_to_model_size()`: Image preprocessing
- `annotate_image()`: Bounding box annotation with predictions

### `scripts/chokepoint.py`
- `load_chokepoint_model()`: Load YOLOv8 fine-tuned model
- `run_chokepoint_inference()`: Run inference on chokepoint model
- Model weights auto-download from GitHub with fallback to yolov8n

### `scripts/fishing.py`
- `get_roboflow_client()`: Lazy-load Roboflow InferenceHTTPClient
- `run_fishing_inference()`: Run inference on Roboflow fishing model

### `scripts/dataframe.py`
- `build_dataframe_row()`: Construct row based on model type
- `create_dataframe()`: Create pandas DataFrame from results

## Live Demo

[https://camera-fishing-effort.fly.dev](https://camera-fishing-effort.fly.dev)

## License

MIT