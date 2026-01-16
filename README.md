# 🎣 Camera Fishing Effort

A FastAPI + Jinja2 web application for uploading fishing camera images, extracting metadata, and preparing data for analysis.

## 🌐 Live Demo

[https://camera-fishing-effort.fly.dev](https://camera-fishing-effort.fly.dev)

## ✨ Features

- **Multiple Image Upload** - Upload several images at once
- **EXIF Extraction** - Automatically extracts date and time from image metadata
- **Location Tagging** - Add location information to your uploads
- **Data Table** - View all uploaded image data in a structured table
- **CSV Export** - Download your data as a CSV file
- **Memory Optimized** - Images are resized for efficient processing

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI (Python) |
| Templating | Jinja2 |
| Frontend | Vanilla JavaScript, CSS |
| Data Processing | Pandas, Pillow |
| Deployment | Fly.io, Docker |
| CI/CD | GitHub Actions |

## 📁 Project Structure

```
camera_fishing_effort/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── fly.toml                # Fly.io deployment config
├── static/
│   └── css/
│       └── style.css       # Styles
├── templates/
│   ├── index.html          # Main upload page
│   └── image_preview.html  # Results display
└── .github/
    └── workflows/
        └── fly-deploy.yml  # Auto-deploy on push
```

## 🚀 Local Development

### Prerequisites
- Python 3.11+
- Conda (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/camera_fishing_effort.git
cd camera_fishing_effort

# Create conda environment
conda create -n camera_fishing_effort python=3.11
conda activate camera_fishing_effort

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 🌍 Deployment

The app automatically deploys to Fly.io when you push to the `main` branch.

### Manual Deployment

```bash
fly deploy
```

## 📊 Data Output

The app generates a CSV with the following columns:

| Column | Description |
|--------|-------------|
| Sl No | Sequential number |
| Image Name | Original filename |
| Location | User-provided location |
| Date | Extracted from EXIF |
| Time | Extracted from EXIF |
| Count | Model inference results (coming soon) |

## 🔮 Roadmap

- [ ] Roboflow model integration for object detection
- [ ] Display annotated images with bounding boxes
- [ ] Batch processing improvements

## 📄 License

MIT