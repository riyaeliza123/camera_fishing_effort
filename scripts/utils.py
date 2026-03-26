"""Utility functions for image processing and data handling."""

import io
import base64
from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS
from io import BytesIO
import numpy as np


def extract_datetime_from_image(image_bytes: bytes) -> tuple:
    """Extract date and time from image EXIF data.
    
    Args:
        image_bytes: Image file as bytes
        
    Returns:
        Tuple of (date_str, time_str) or ("N/A", "N/A") if not found
    """
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


def stretch_image_to_model_size(image_bytes: bytes, target_size: int = 640) -> tuple:
    """Stretch image to model input size and save to temp file.
    
    Args:
        image_bytes: Image file as bytes
        target_size: Target size (default 640x640)
        
    Returns:
        Tuple of (temp_file_path, PIL_Image)
    """
    img_pil = Image.open(BytesIO(image_bytes))
    img_stretched = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Generate temp filename from original if available
    temp_path = "temp_inference.jpg"
    img_stretched.save(temp_path, format="JPEG")
    
    return temp_path, img_stretched


def annotate_image(temp_path: str, pred_boxes: list, pred_conf: list, 
                   model_type: str, pred_classes: list = None) -> str:
    """Annotate image with bounding boxes and predictions.
    
    Args:
        temp_path: Path to the image file
        pred_boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        pred_conf: List of confidence scores
        model_type: "chokepoint" or "fishing"
        pred_classes: List of class indices (for chokepoint model)
        
    Returns:
        Base64 encoded image data URL
    """
    # Use PIL drawing instead of matplotlib to avoid native backend issues in containers.
    img_pil = Image.open(temp_path).convert("RGB")
    draw = ImageDraw.Draw(img_pil)
    
    class_names_map = {
        "chokepoint": ['in', 'out'],
        "fishing": ['boat']
    }
    class_names_local = class_names_map.get(model_type, ['object'])
    
    for i, (box, conf) in enumerate(zip(pred_boxes, pred_conf)):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
        
        if model_type == "chokepoint" and pred_classes is not None:
            class_idx = int(pred_classes[i]) if i < len(pred_classes) else 0
            class_idx = class_idx if 0 <= class_idx < len(class_names_local) else 0
            label = f"{class_names_local[class_idx]} ({float(conf):.2f})"
        else:
            label = f"boat ({float(conf):.2f})"

        text_x = max(0, int(x1))
        text_y = max(0, int(y1) - 14)
        draw.rectangle((text_x, text_y, text_x + 170, text_y + 14), fill="black")
        draw.text((text_x + 2, text_y + 1), label, fill="white")
    
    # Save annotated image to buffer
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    buf.seek(0)
    base64_data = base64.b64encode(buf.read()).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{base64_data}"
    
    return data_url
