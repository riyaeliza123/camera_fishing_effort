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

## Setup Instructions

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your Roboflow API key
   # .env should look like:
   # ROBOFLOW_API_KEY=your-api-key-here
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

## Environment Variables

- `ROBOFLOW_API_KEY`: Your Roboflow API key for the fishing model (required for fishing model)

## Live Demo

[https://camera-fishing-effort.fly.dev](https://camera-fishing-effort.fly.dev)

## License

MIT