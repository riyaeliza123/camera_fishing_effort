# Fishing Effort Monitoring - Computer Vision Model
A FastAPI web application for analyzing fishing camera images using AI-powered object detection. Upload images, automatically detect boats, and export results as CSV.

## 🚀 Live Application

**Visit the Live App**: https://camera-fishing-effort.fly.dev

## Overview

This project combines YOLOv8 deep learning models with a web interface to detect and count boats in fishing camera footage. It supports two detection modes:

- **Chokepoint Mode**: Detects boats entering/exiting a specific location (In/Out counts)
- **Fishing Mode**: Detects boats using Roboflow's fishing-access-points model

### Key Features

- Upload single or multiple images
- Real-time object detection with annotated results
- Automatic EXIF metadata extraction (date/time)
- CSV export of detection results
- Containerized with Docker for production deployment
- Live on Fly.io with no setup required

## Documentation

For detailed information, see the documentation folder:

- **[workflow.md](documentation/workflow.md)** - How to use the app and understand the data flow
- **[YOLO.md](documentation/YOLO.md)** - Computer Vision, Neural networks, YOLOv8, and object detection explained
- **[functions.md](documentation/functions.md)** - Project architecture and code organization
- **[definitions_and_metrics.md](documentation\definitions_and_metrics.md)** - Defines terms, metrics and goals. Clarifies metrics and definition of success.

## Quick Start

**Local Development:**
```bash
pip install -r requirements.txt
python main.py
# Visit http://127.0.0.1:8000
```

**Production (Fly.io):**
```bash
fly secrets set ROBOFLOW_API_KEY="your-key"
fly deploy
```

---

For complete setup instructions, features, and technical details, see the documentation folder.

## License

MIT