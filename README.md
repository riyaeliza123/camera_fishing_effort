# Camera Fishing Effort - Image Upload App

A FastAPI + HTMX application for uploading and analyzing fishing images. This app will eventually integrate with Roboflow for model inference.

## 🚀 Features

- **FastAPI Backend**: Modern, fast Python web framework
- **HTMX Frontend**: Dynamic interactions without writing JavaScript
- **Image Upload**: Simple drag-and-drop or click to upload
- **Responsive Design**: Beautiful UI that works on all devices

## 📋 Planned Features

- Multiple image upload support
- Roboflow Hosted Inference integration
- Deployment to fly.io

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🏃 Running the App

Start the development server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload
```

The app will be available at: **http://localhost:8000**

## 📁 Project Structure

```
camera_fishing_effort/
├── main.py                 # FastAPI application
├── templates/              # HTML templates
│   ├── index.html         # Main upload page
│   └── image_preview.html # Image preview component
├── static/
│   ├── css/
│   │   └── style.css      # Styles
│   └── uploads/           # Uploaded images (auto-created)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔮 Next Steps

1. Add multiple image upload support
2. Integrate Roboflow API for model inference
3. Display detection results
4. Deploy to fly.io

## 📝 Notes

- Uploaded images are stored in `static/uploads/`
- Each upload gets a unique UUID filename
- HTMX handles dynamic content updates without page reloads