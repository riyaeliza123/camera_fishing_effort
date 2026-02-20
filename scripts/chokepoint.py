"""Chokepoint model inference and processing."""

import os
import urllib.request
from ultralytics import YOLO
from scripts.constants import MODEL_PATH, GITHUB_MODEL_URL, CHOKEPOINT_CONFIDENCE_THRESHOLD, CHOKEPOINT_CLASS_NAMES


# Global model variable
model_trained = None


def download_model_weights(url: str, local_path: str) -> bool:
    """Download model weights from GitHub.
    
    Args:
        url: GitHub URL to model weights
        local_path: Local path to save weights
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading model weights from {url}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Model weights downloaded successfully to {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading model weights: {e}")
        return False


def load_chokepoint_model():
    """Load the fine-tuned chokepoint model.
    
    Returns:
        YOLO model instance
    """
    global model_trained
    
    if model_trained is not None:
        return model_trained
    
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
        return model_trained
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}. Falling back to default yolov8n model.")
        model_trained = YOLO('yolov8n.pt')
        return model_trained


def run_chokepoint_inference(temp_path: str) -> dict:
    """Run inference on chokepoint model.
    
    Args:
        temp_path: Path to the stretched image
        
    Returns:
        Dictionary with predictions: {
            'pred_classes': array,
            'pred_boxes': array,
            'pred_conf': array,
            'in_count': int,
            'out_count': int,
            'class_names': list
        }
    """
    model = load_chokepoint_model()
    results = model(temp_path)
    
    pred_classes = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes is not None else []
    pred_boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
    pred_conf = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
    
    # Filter by confidence threshold
    filtered_indices = [i for i in range(len(pred_conf)) if pred_conf[i] >= CHOKEPOINT_CONFIDENCE_THRESHOLD]
    pred_classes = pred_classes[filtered_indices] if len(filtered_indices) > 0 else []
    pred_boxes = pred_boxes[filtered_indices] if len(filtered_indices) > 0 else []
    pred_conf = pred_conf[filtered_indices] if len(filtered_indices) > 0 else []
    
    # Count 'in' and 'out'
    in_count = sum(1 for c in pred_classes if c == 0)
    out_count = sum(1 for c in pred_classes if c == 1)
    
    return {
        'pred_classes': pred_classes,
        'pred_boxes': pred_boxes,
        'pred_conf': pred_conf,
        'in_count': in_count,
        'out_count': out_count,
        'boat_count': None,
        'class_names': CHOKEPOINT_CLASS_NAMES
    }
