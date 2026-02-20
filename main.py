from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import List
from io import BytesIO
import os
import base64

# Import modularized functions
from scripts.config import load_config, get_roboflow_api_key
from scripts.utils import extract_datetime_from_image, stretch_image_to_model_size, annotate_image
from scripts.chokepoint import run_chokepoint_inference
from scripts.fishing import run_fishing_inference
from scripts.dataframe import build_dataframe_row, create_dataframe
from scripts.constants import DEFAULT_IMAGE_SIZE

import pandas as pd

app = FastAPI()

# Load configuration
config = load_config()
roboflow_api_key = get_roboflow_api_key(config)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Store the latest DataFrame for CSV download
latest_df = None


@app.get("/")
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
                
                # Stretch image to model size
                temp_path, _ = stretch_image_to_model_size(contents, DEFAULT_IMAGE_SIZE)
                
                # Run inference based on model type
                if model_type == "chokepoint":
                    inference_result = run_chokepoint_inference(temp_path)
                else:  # fishing
                    inference_result = run_fishing_inference(temp_path, roboflow_api_key)
                
                # Extract results
                pred_classes = inference_result['pred_classes']
                pred_boxes = inference_result['pred_boxes']
                pred_conf = inference_result['pred_conf']
                in_count = inference_result['in_count']
                out_count = inference_result['out_count']
                boat_count = inference_result['boat_count']
                
                # Annotate image
                data_url = annotate_image(temp_path, pred_boxes, pred_conf, model_type, pred_classes)
                
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
                row = build_dataframe_row(
                    idx, img.filename, location, image_date, image_time,
                    model_type, in_count, out_count, boat_count
                )
                df_data.append(row)
                
                # Clean up temp file
                os.remove(temp_path)
        
        latest_df = create_dataframe(df_data)
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
