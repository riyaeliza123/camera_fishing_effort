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

app = FastAPI()

# Load YOLOv8 model weights once
MODEL_PATH = "notebooks/runs/detect/chokepoint_finetuned/train/weights/best.pt"
model_trained = YOLO(MODEL_PATH)
class_names = ['in', 'out']

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
    files: list = File(...)
):
    """Handle multiple image uploads, run YOLOv8 inference, annotate images, and return table with counts."""
    global latest_df
    try:
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
                # Run YOLOv8 inference
                results = model_trained(temp_path)
                pred_classes = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes is not None else []
                # Count 'in' and 'out'
                in_count = sum(1 for c in pred_classes if c == 0)
                out_count = sum(1 for c in pred_classes if c == 1)
                # Annotate image
                img_pil = Image.open(temp_path)
                plt.figure(figsize=(6,6))
                plt.imshow(img_pil)
                ax = plt.gca()
                pred_boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
                for box, cls in zip(pred_boxes, pred_classes):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='lime', linewidth=1)
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 2, class_names[cls], color='white', fontsize=8, va='bottom', ha='left', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=0))
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
                # Build DataFrame row
                df_data.append({
                    "Sl No": idx,
                    "Image Name": img.filename,
                    "Location": location,
                    "Date": image_date,
                    "Time": image_time,
                    "In": in_count,
                    "Out": out_count
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
