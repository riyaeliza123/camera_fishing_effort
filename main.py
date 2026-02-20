from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import StreamingResponse
import io
import base64
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import os
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import base64
from typing import List
import pandas as pd
from io import BytesIO
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import urllib.request
from inference_sdk import InferenceHTTPClient
import tomllib
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()


# Get Roboflow API key from environment
def _get_roboflow_api_key():
    """Get Roboflow API key from environment"""
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("Warning: ROBOFLOW_API_KEY not found in environment. Fishing access point model will not be available.")
    return api_key

# Download model weights from GitHub if not available locally
MODEL_PATH = "best.pt"
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/riyaeliza123/camera_fishing_effort/main/notebooks/runs/detect/chokepoint_finetuned/train/weights/best.pt"

def download_model_weights(url: str, local_path: str) -> bool:
    """Download model weights from GitHub"""
    try:
        print(f"Downloading model weights from {url}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Model weights downloaded successfully to {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading model weights: {e}")
        return False

# Load YOLOv8 model weights once
model_trained = None
try:
    # Try to load local model, if not found, download from GitHub
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found locally. Attempting to download from GitHub...")
        if download_model_weights(GITHUB_MODEL_URL, MODEL_PATH):
            model_trained = YOLO(MODEL_PATH)
        else:
            raise Exception("Failed to download model from GitHub")
    else:
        model_trained = YOLO(MODEL_PATH)
    print("Fine-tuned YOLOv8 model loaded successfully")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}. Falling back to default yolov8n model.")
    model_trained = YOLO('yolov8n.pt')
class_names = ['in', 'out']

# Roboflow client setup (lazy-load)
roboflow_client = None
roboflow_api_key = _get_roboflow_api_key()

def get_roboflow_client():
    """Lazy-load Roboflow client"""
    global roboflow_client
    if roboflow_client is None:
        if not roboflow_api_key:
            raise RuntimeError("Roboflow API key not configured. Please set ROBOFLOW_API_KEY environment variable.")
        roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=roboflow_api_key
        )
        print("Roboflow client initialized successfully")
    return roboflow_client

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Store the latest DataFrame for CSV download
latest_df = None


def extract_datetime_from_image(image_bytes: bytes) -> tuple:
    """Extract date and time from image EXIF data, returns (date, time)"""
    try:
        img = Image.open(BytesIO(image_bytes))
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTimeOriginal" or tag == "DateTime":
                    # EXIF format: "YYYY:MM:DD HH:MM:SS"
                    parts = value.split(" ")
                    if len(parts) == 2:
                        date_str = parts[0].replace(":", "-")  # Convert to YYYY-MM-DD
                        time_str = parts[1]
                        return (date_str, time_str)
                    return (value, "N/A")
    except Exception as e:
        print(f"Error extracting EXIF: {e}")
    return ("N/A", "N/A")


def resize_image_for_display(image_bytes: bytes, max_size: int = 800) -> bytes:
    """Resize image for display to reduce memory usage"""
    try:
        img = Image.open(BytesIO(image_bytes))
        
        # Calculate new size maintaining aspect ratio
        ratio = min(max_size / img.width, max_size / img.height)
        if ratio < 1:  # Only resize if image is larger than max_size
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary (for PNG with transparency)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Save to bytes
        output = BytesIO()
        img.save(output, format='JPEG', quality=85)
        return output.getvalue()
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image_bytes  # Return original if resize fails


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/test")
async def test():
    """Simple test endpoint"""
    return {"status": "ok"}



@app.post("/upload")
async def upload_images(
    request: Request,
    location: str = Form(...),
    model_type: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Handle multiple image uploads, run inference based on model type, annotate images, and return table with counts."""
    global latest_df
    
    # Set confidence threshold based on model type
    CONFIDENCE_THRESHOLD = 0.7 if model_type == "fishing" else 0.5
    
    try:
        # Validate model type
        if model_type not in ["chokepoint", "fishing"]:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid model type. Choose 'chokepoint' or 'fishing'."}
            )
        
        # Check Roboflow API key for fishing model
        if model_type == "fishing" and not roboflow_api_key:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={"error": "Fishing model is not available. Roboflow API key not configured."}
            )
        
        uploaded_images = []
        df_data = []
        for idx, img in enumerate(files, start=1):
            if img.filename:
                contents = await img.read()
                # Stretch image to 640x640 before inference
                img_pil = Image.open(BytesIO(contents))
                img_stretched = img_pil.resize((640, 640), Image.Resampling.LANCZOS)
                temp_path = f"temp_{img.filename}"
                img_stretched.save(temp_path, format="JPEG")
                
                # Run inference based on model type
                if model_type == "chokepoint":
                    results = model_trained(temp_path)
                    pred_classes = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes is not None else []
                    pred_boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
                    pred_conf = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
                    
                    # Filter by confidence threshold
                    filtered_indices = [i for i in range(len(pred_conf)) if pred_conf[i] >= CONFIDENCE_THRESHOLD]
                    pred_classes = pred_classes[filtered_indices] if len(filtered_indices) > 0 else []
                    pred_boxes = pred_boxes[filtered_indices] if len(filtered_indices) > 0 else []
                    pred_conf = pred_conf[filtered_indices] if len(filtered_indices) > 0 else []
                    
                    # Count 'in' and 'out'
                    in_count = sum(1 for c in pred_classes if c == 0)
                    out_count = sum(1 for c in pred_classes if c == 1)
                    boat_count = None
                    class_names_local = ['in', 'out']
                    
                else:  # fishing model
                    client = get_roboflow_client()
                    result = client.infer(temp_path, model_id="fishing-access-points/2")
                    
                    # Extract predictions from Roboflow result
                    pred_boxes = []
                    pred_conf = []
                    if "predictions" in result:
                        for pred in result["predictions"]:
                            # Extract bounding box coordinates
                            x = pred.get("x")
                            y = pred.get("y")
                            width = pred.get("width")
                            height = pred.get("height")
                            confidence = pred.get("confidence", 0)
                            
                            if confidence >= CONFIDENCE_THRESHOLD:
                                x1 = x - width / 2
                                y1 = y - height / 2
                                x2 = x + width / 2
                                y2 = y + height / 2
                                pred_boxes.append([x1, y1, x2, y2])
                                pred_conf.append(confidence)
                    
                    in_count = None
                    out_count = None
                    boat_count = len(pred_boxes)
                    class_names_local = ['boat']
                
                # Annotate image
                img_pil = Image.open(temp_path)
                plt.figure(figsize=(6, 6))
                plt.imshow(img_pil)
                ax = plt.gca()
                
                for i, (box, conf) in enumerate(zip(pred_boxes, pred_conf)):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='lime', linewidth=1)
                    ax.add_patch(rect)
                    
                    if model_type == "chokepoint":
                        # For chokepoint, we need class index
                        class_idx = pred_classes[i] if i < len(pred_classes) else 0
                        label = f"{class_names_local[class_idx]} ({conf:.2f})"
                    else:
                        label = f"boat ({conf:.2f})"
                    
                    ax.text(x1, y1 - 2, label, color='white', fontsize=8, va='bottom', ha='left',
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=0))
                
                plt.axis('off')
                plt.title(f'Predictions: {os.path.basename(temp_path)}')
                # Save annotated image to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
                plt.close()
                buf.seek(0)
                base64_data = base64.b64encode(buf.read()).decode('utf-8')
                data_url = f"data:image/jpeg;base64,{base64_data}"
                uploaded_images.append({
                    "url": data_url,
                    "filename": img.filename
                })
                # Extract date and time from EXIF (with fallback)
                try:
                    image_date, image_time = extract_datetime_from_image(contents)
                except Exception as e:
                    print(f"EXIF extraction failed for {img.filename}: {e}")
                    image_date, image_time = "N/A", "N/A"
                # Build DataFrame row (different columns based on model type)
                if model_type == "chokepoint":
                    df_data.append({
                        "Sl No": idx,
                        "Image Name": img.filename,
                        "Location": location,
                        "Date": image_date,
                        "Time": image_time,
                        "In": in_count,
                        "Out": out_count
                    })
                else:  # fishing model
                    df_data.append({
                        "Sl No": idx,
                        "Image Name": img.filename,
                        "Location": location,
                        "Date": image_date,
                        "Time": image_time,
                        "Boat": boat_count
                    })
                # Clean up temp file
                os.remove(temp_path)
        latest_df = pd.DataFrame(df_data)
        return templates.TemplateResponse(
            "image_preview.html",
            {
                "request": request,
                "images": uploaded_images,
                "count": len(uploaded_images),
                "location": location,
                "model_type": model_type,
                "df_records": latest_df.to_dict('records'),
                "df_columns": latest_df.columns.tolist()
            }
        )
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/download-csv")
async def download_csv():
    """Download the latest DataFrame as CSV"""
    global latest_df
    
    if latest_df is None or latest_df.empty:
        return {"error": "No data available"}
    
    # Create CSV in memory
    csv_buffer = BytesIO()
    latest_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return StreamingResponse(
        csv_buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=image_data.csv"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
