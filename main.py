from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import base64
from typing import List

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/test")
async def test():
    """Simple test endpoint"""
    return {"status": "ok"}


@app.post("/upload")
async def upload_images(request: Request, files: List[UploadFile] = File(...)):
    """Handle multiple image uploads and return the image previews"""
    print(f"Received {len(files)} files")  # Debug log
    
    uploaded_images = []
    
    for img in files:
        if img.filename:
            # Read file content and convert to base64 data URL
            contents = await img.read()
            base64_data = base64.b64encode(contents).decode('utf-8')
            content_type = img.content_type or 'image/jpeg'
            data_url = f"data:{content_type};base64,{base64_data}"
            
            print(f"  - {img.filename}, size: {len(contents)} bytes")
            
            uploaded_images.append({
                "url": data_url,
                "filename": img.filename
            })
    
    return templates.TemplateResponse(
        "image_preview.html",
        {
            "request": request,
            "images": uploaded_images,
            "count": len(uploaded_images)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
