from fastapi import FastAPI, File, UploadFile, Request, Form
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
    files: List[UploadFile] = File(...)
):
    """Handle multiple image uploads and return the image previews"""
    global latest_df
    
    try:
        print(f"Location: {location}")  # Debug log
        print(f"Received {len(files)} files")
        
        uploaded_images = []
        df_data = []
        
        for idx, img in enumerate(files, start=1):
            if img.filename:
                # Read file content
                contents = await img.read()
                
                # Extract date and time from EXIF (with fallback)
                try:
                    image_date, image_time = extract_datetime_from_image(contents)
                except Exception as e:
                    print(f"EXIF extraction failed for {img.filename}: {e}")
                    image_date, image_time = "N/A", "N/A"
                
                # Resize image for display to reduce memory
                resized_contents = resize_image_for_display(contents)
                
                # Convert to base64 data URL for display
                base64_data = base64.b64encode(resized_contents).decode('utf-8')
                data_url = f"data:image/jpeg;base64,{base64_data}"
                
                # Free up memory
                del contents
                del resized_contents
                
                print(f"  - {img.filename}, date: {image_date}, time: {image_time}")
                
                uploaded_images.append({
                    "url": data_url,
                    "filename": img.filename
                })
                
                # Build DataFrame row
                df_data.append({
                    "Sl No": idx,
                    "Image Name": img.filename,
                    "Location": location,
                    "Date": image_date,
                    "Time": image_time,
                    "Count": "Pending"  # Placeholder for Roboflow inference
                })
        
        # Create DataFrame
        latest_df = pd.DataFrame(df_data)
        print("\nDataFrame:")
        print(latest_df.to_string())
        
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
