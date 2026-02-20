# Camera Fishing Effort - Workflow Documentation

## User Perspective: How to Use the App

### Step 1: Access the Application
Visit the live application at [https://camera-fishing-effort.fly.dev](https://camera-fishing-effort.fly.dev)

You'll see a clean interface with three main input fields:

### Step 2: Enter Location
- Click on the **"📍 Location"** field
- Type in the fishing location (e.g., "Bear Cove", "Duval South")
- This metadata helps identify where the images were captured

### Step 3: Select Detection Model
- Click on the **"🤖 Detection Model"** dropdown
- Choose one of two options:
  - **Chokepoint (In/Out Detection)**: Detects boats entering/exiting a specific location
  - **Fishing (Boat Detection)**: Detects boats using the Roboflow fishing-access-points model
- **Note**: The upload button is disabled until you select a model

### Step 4: Upload Images
- Click on the **"Choose images"** upload area or drag-and-drop image files
- You can upload **multiple images at once** or **one at a time**
- Supported formats: JPEG, PNG, and other common image formats
- Click **"Upload Images"** button to process

### Step 5: View Results
The app processes your images and displays:

#### **Annotated Images**
- Your uploaded images with **green bounding boxes** around detected objects
- Each box shows:
  - The object class (e.g., "in", "out", or "boat")
  - Confidence score (0.55 = 55%)

#### **Results Table**
The table structure depends on your selected model:

**For Chokepoint Model:**
| Sl No | Image Name | Location | Date | Time | In | Out |
|-------|------------|----------|------|------|----|----|
| 1 | image1.jpg | Bear Cove | 2026-02-20 | 14:32:15 | 2 | 1 |

**For Fishing Model:**
| Sl No | Image Name | Location | Date | Time | Boat |
|-------|------------|----------|------|------|------|
| 1 | image1.jpg | Duval South | 2026-02-20 | 14:32:15 | 3 |

#### **Image Metadata**
- **Date & Time**: Automatically extracted from image EXIF data (camera timestamp)
- Falls back to "N/A" if no EXIF data is available

### Step 6: Download Results as CSV
- Click **"📥 Download as CSV"** button
- A CSV file downloads with all results from the current session
- Open in Excel, Google Sheets, or any spreadsheet application for further analysis

### Step 7: Continue Uploading (Optional)
- Click **"Upload More Images"** to process additional images
- You can upload new images to the same location or change location/model
- All results accumulate in the table until you refresh the page

---

## Data Flow: Detailed Processing Pipeline

### 1. Data Ingestion

```
User Uploads Images
         ↓
Browser sends multipart/form-data to /upload endpoint
         ↓
FastAPI receives:
  - location (string)
  - model_type (string: "chokepoint" or "fishing")
  - files (List[UploadFile])
```

**What happens:**
- Each file is read from memory (not saved to disk)
- File metadata is preserved (filename, etc.)
- Images are validated to ensure they're readable

### 2. Image Preprocessing

```python
For each image:
  1. Read bytes from upload
  2. Open with PIL (Python Imaging Library)
  3. Resize to 640x640 pixels (model input size)
     - Uses LANCZOS resampling for high quality
     - Maintains aspect ratio by stretching
  4. Save as JPEG to temporary file (temp_inference.jpg)
```

**Why 640x640?**
- Both YOLOv8 and Roboflow models expect 640x640 input
- Standardized size ensures consistent model performance
- Temporary file used for inference, then immediately deleted

### 3. Model Inference (Two Paths)

#### **Path A: Chokepoint Model (YOLOv8)**

```
Image File (temp_inference.jpg)
         ↓
Load Fine-tuned YOLOv8 Model
  - First load: Downloads from GitHub if not local
  - Subsequent loads: Uses cached model in memory
         ↓
Run Inference
  - YOLOv8 processes image at 640x640
  - Returns predictions for each detected object
         ↓
Extract Predictions
  - Class (0=in, 1=out)
  - Bounding box coordinates (x1,y1,x2,y2)
  - Confidence score (0.0-1.0)
         ↓
Filter by Confidence Threshold (0.5)
  - Only keep predictions with confidence ≥ 0.5
  - Removes weak detections
         ↓
Count Objects
  - Sum of objects with class=0 → "In" count
  - Sum of objects with class=1 → "Out" count
```

**Model Details:**
- Fine-tuned on chokepoint fishing data
- Fallback: If download fails, uses YOLOv8n (nano, less accurate)
- Lazy-loaded: Model loaded only on first use, then cached

#### **Path B: Fishing Model (Roboflow)**

```
Image File (temp_inference.jpg)
         ↓
Initialize Roboflow Client (lazy-load)
  - Uses API key from config.toml
  - Creates InferenceHTTPClient
  - Connects to: https://detect.roboflow.com
         ↓
Send Image to Roboflow API
  - HTTP POST request with image file
  - Model ID: fishing-access-points/2
  - Returns: JSON with predictions
         ↓
Parse API Response
  - Extract predicted objects
  - For each prediction:
    - Get center coordinates (x, y)
    - Get box dimensions (width, height)
    - Get confidence score
         ↓
Convert to Standard Format
  - Convert from center format to corner format
  - x1 = x - width/2
  - y1 = y - height/2
  - x2 = x + width/2
  - y2 = y + height/2
         ↓
Filter by Confidence Threshold (0.7)
  - Only keep predictions with confidence ≥ 0.7
  - Higher threshold: fewer false positives
         ↓
Count Objects
  - Total boats = number of boxes that passed threshold
```

**API Details:**
- Real-time inference via Roboflow API
- Requires valid API key in config.toml
- Lazy-loaded: Client created only on first use

### 4. Image Annotation & Visualization

```
For each detected object:
  1. Get bounding box coordinates
  2. Load original image (temp_inference.jpg)
  3. Create matplotlib figure
  4. Draw rectangle with coordinates
  5. Add label with class name and confidence
     - Chokepoint: Shows "in (0.95)" or "out (0.87)"
     - Fishing: Shows "boat (0.92)"
  6. Convert figure to JPEG
  7. Encode as base64
  8. Return as data URL for HTML display
```

**Visual Output:**
- Green bounding boxes on original image
- Black background labels with white text
- Displayed as embedded image in results page (no file download needed)

### 5. Metadata Extraction

```
Original Image File (raw bytes)
         ↓
Extract EXIF Data
  - Look for DateTimeOriginal or DateTime tags
  - Format: "YYYY:MM:DD HH:MM:SS"
         ↓
Parse into Components
  - Date: Convert "2026-02-20" format
  - Time: Extract "14:32:15"
         ↓
If EXIF not found → Return "N/A"
```

**EXIF (Exchangeable Image File Format):**
- Metadata embedded in image by camera
- Timestamp represents when photo was taken
- Essential for tracking when fishing activity occurred

### 6. DataFrame Construction

```
Collected Data for Each Image:
  - Index (1, 2, 3, ...)
  - Filename
  - Location (user input)
  - Date (from EXIF)
  - Time (from EXIF)
  - Model-specific counts
         ↓
Create Row Dictionary
  If Chokepoint:
    {"Sl No": 1, "Image Name": "fish1.jpg", ..., "In": 2, "Out": 1}
  
  If Fishing:
    {"Sl No": 1, "Image Name": "fish1.jpg", ..., "Boat": 3}
         ↓
Aggregate All Rows
  - Collect rows from all processed images
  - Create pandas DataFrame
         ↓
Convert to Display Format
  - .to_dict('records') for HTML table
  - .columns.tolist() for header row
```

### 7. Response Generation

```
Collected Results:
  - List of annotated image data URLs
  - DataFrame as dictionary records
  - DataFrame column names
  - Original location
  - Selected model type
         ↓
Render HTML Template
  - image_preview.html receives context
  - Jinja2 templating engine processes:
    * Loop through images and display each
    * Conditionally show columns based on model_type
    * Display In/Out for chokepoint, Boat for fishing
         ↓
Return to Browser
  - HTTP 200 response with HTML
  - Images embedded as base64
  - Interactive table with results
```

### 8. CSV Export

```
User Clicks "Download as CSV"
         ↓
Access latest_df (pandas DataFrame in memory)
         ↓
Convert to CSV Format
  - Headers from column names
  - Data rows from each image result
         ↓
Encode as Bytes
  - Create BytesIO buffer
  - Write CSV content to buffer
         ↓
Stream to Browser
  - HTTP response with CSV content type
  - Filename: image_data.csv
  - Browser downloads file automatically
         ↓
User Opens in Excel/Sheets
  - Rows: one per image
  - Columns: location, date, time, counts
```

---

## Containerization & Deployment

### Docker: Containerizing the Application

**What is Docker?**

Docker is a containerization platform that packages this entire application into a lightweight, isolated container. Think of it as a self-contained box containing:
- The application code
- All dependencies (Python packages)
- System libraries
- Configuration
- Everything needed to run the app

**Why Docker?**

1. **Consistency**: Works the same on our laptop, team's computers, and production servers
2. **Isolation**: Dependencies don't conflict with other projects
3. **Portability**: Move container between any system
4. **Reproducibility**: Anyone can run the exact same environment

**Our Dockerfile Process**

```dockerfile
FROM python:3.11-slim-bullseye
├─ Start with official Python 3.11 base image
├─ "slim-bullseye" = lightweight Linux OS

WORKDIR /app
└─ Set working directory inside container

RUN apt-get install libgl1 libglib2.0-0
└─ Install system libraries (needed for image processing)

COPY requirements.txt .
COPY constraints.txt .
└─ Copy dependency definitions

RUN pip install -r requirements.txt
└─ Install all Python packages

COPY main.py .
COPY scripts/ ./scripts/
COPY static/ ./static/
COPY templates/ ./templates/
└─ Copy application code

ENV ROBOFLOW_API_KEY=""
└─ Prepare for API key injection at runtime

EXPOSE 8080
└─ Container listens on port 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
└─ Start the FastAPI application
```

**Build & Run Locally (if interested in development):**
```bash
# Build the container image
docker build -t camera-fishing-effort .

# Run the container
docker run -p 8000:8080 \
  -e ROBOFLOW_API_KEY=your-key \
  camera-fishing-effort

# Visit http://localhost:8000
```

### Fly.io: Cloud Deployment

**What is Fly.io?**

Fly.io is a Platform-as-a-Service (PaaS) that:
- Takes this Docker container
- Deploys it globally across data centers
- Provides automatic scaling, SSL certificates, and monitoring
- Manages the infrastructure of the application

**Why Fly.io?**

1. **Live Demo**: App is publicly accessible at https://camera-fishing-effort.fly.dev
2. **No Hardware**: Don't need to maintain a server
3. **Always Running**: 24/7 availability without manual restart
4. **Global CDN**: Distributed across regions for fast access
5. **Easy Updates**: Deploy new versions with one command

**Deployment Workflow**

```
Local Development (main.py + scripts/)
         ↓
Commit to GitHub
         ↓
Build Docker Image
  - Fly.io pulls from GitHub
  - Runs docker build
  - Creates container
         ↓
Deploy to Fly.io
  - Container pushed to data center
  - Network configured
  - SSL certificate created
  - Health checks enabled
         ↓
Public URL: https://camera-fishing-effort.fly.dev
  - App accessible from anywhere
  - No local machine running
```

**Configuration: fly.toml**

```toml
[env]
ULTRALYTICS_SETTINGS=false
ROBOFLOW_API_KEY=""    ← Set via: fly secrets set ROBOFLOW_API_KEY="..."

[http_service]
internal_port = 8080      # Container port
force_https = true        # Encrypt all traffic
auto_start_machines = true
min_machines_running = 0  # Scale down when idle
```

**Deploy New Version:**
```bash
# Set secrets (one-time)
fly secrets set ROBOFLOW_API_KEY="your-key"

# Deploy updated code
fly deploy

# View logs
fly logs
```

### End-to-End Workflow

```
┌────────────────────────────────────────────────────────────┐
│ DEVELOPMENT (Local Machine)                                │
│                                                            │
│ 1. Write code (main.py, scripts/)                          │
│ 2. Test locally: python main.py                            │
│ 3. Build Docker: docker build -t app .                     │
│ 4. Test in container: docker run -p 8000:8080 app          │
│ 5. Commit to GitHub: git push                              │
└────────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────────┐
│ PRODUCTION (Fly.io Cloud)                                 │
│                                                           │
│ 1. Fly.io detects push to GitHub                          │
│ 2. Pulls code, builds Docker image                        │
│ 3. Deploys container globally                             │
│ 4. App live at: https://camera-fishing-effort.fly.dev     │
│                                                           │
│ Users can now:                                            │
│ - Access app without running code                         │
│ - Upload images anytime                                   │
│ - Get instant results                                     │
│ - Share link without explaining setup                     │
└───────────────────────────────────────────────────────────┘
```

### Benefits of This Setup

**For Development:**
- Changes made locally are immediately testable
- No need to manage server infrastructure
- Easy rollback if something breaks

**For Demos & Collaboration:**
- Share single URL instead of setup instructions
- Works on any device with internet
- No installation required for viewers
- Live updates available instantly

**For Production:**
- Scales automatically with demand
- Built-in monitoring and logging

---

## Workflow summary

**User Experience:**
- Simple 3-step workflow: Select -> Upload -> Download 
- Real-time results with annotated images
- Works with multiple images seamlessly

**Data Processing:**
- Sophisticated pipeline handling image preprocessing, inference, annotation
- Two different models with different output formats
- Metadata extraction for comprehensive data tracking

**Deployment:**
- Docker ensures consistency across environments
- Fly.io provides global accessibility without infrastructure burden
- Live demo accessible to anyone with a link
