"""Utility functions for image processing and data handling."""

import io
import base64
from PIL import Image
from PIL.ExifTags import TAGS
from io import BytesIO
import matplotlib.pyplot as plt
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
    img_pil = Image.open(temp_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_pil)
    ax = plt.gca()
    
    class_names_map = {
        "chokepoint": ['in', 'out'],
        "fishing": ['boat']
    }
    class_names_local = class_names_map.get(model_type, ['object'])
    
    for i, (box, conf) in enumerate(zip(pred_boxes, pred_conf)):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='lime', linewidth=1)
        ax.add_patch(rect)
        
        if model_type == "chokepoint" and pred_classes is not None:
            class_idx = pred_classes[i] if i < len(pred_classes) else 0
            label = f"{class_names_local[class_idx]} ({conf:.2f})"
        else:
            label = f"boat ({conf:.2f})"
        
        ax.text(x1, y1 - 2, label, color='white', fontsize=8, va='bottom', ha='left',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=0))
    
    plt.axis('off')
    plt.title(f'Predictions')
    
    # Save annotated image to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    base64_data = base64.b64encode(buf.read()).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{base64_data}"
    
    return data_url
