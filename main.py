from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from pathlib import Path
import uuid

app = FastAPI()

# Create necessary directories
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_image(request: Request, image: UploadFile = File(...)):
    """Handle image upload and return the image preview"""
    # Generate unique filename
    file_extension = Path(image.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    # Save the uploaded file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    # Return HTML fragment with the uploaded image
    image_url = f"/static/uploads/{unique_filename}"
    return templates.TemplateResponse(
        "image_preview.html",
        {
            "request": request,
            "image_url": image_url,
            "filename": image.filename
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
