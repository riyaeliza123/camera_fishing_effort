"""Fishing model inference and processing using Roboflow."""

from inference_sdk import InferenceHTTPClient
from scripts.constants import FISHING_CONFIDENCE_THRESHOLD, FISHING_CLASS_NAMES, ROBOFLOW_MODEL_ID, ROBOFLOW_API_URL


# Global Roboflow client variable
roboflow_client = None


def get_roboflow_client(api_key: str) -> InferenceHTTPClient:
    """Get or initialize Roboflow client (lazy-load).
    
    Args:
        api_key: Roboflow API key
        
    Returns:
        InferenceHTTPClient instance
        
    Raises:
        RuntimeError: If API key is not configured
    """
    global roboflow_client
    
    if roboflow_client is None:
        if not api_key:
            raise RuntimeError("Roboflow API key not configured. Please set it in config.toml.")
        roboflow_client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=api_key
        )
        print("Roboflow client initialized successfully")
    return roboflow_client


def run_fishing_inference(temp_path: str, api_key: str) -> dict:
    """Run inference on fishing model via Roboflow.
    
    Args:
        temp_path: Path to the stretched image
        api_key: Roboflow API key
        
    Returns:
        Dictionary with predictions: {
            'pred_classes': list (empty for fishing model),
            'pred_boxes': list,
            'pred_conf': list,
            'in_count': None,
            'out_count': None,
            'boat_count': int,
            'class_names': list
        }
        
    Raises:
        HTTPCallErrorError: If Roboflow API call fails
    """
    client = get_roboflow_client(api_key)
    result = client.infer(temp_path, model_id=ROBOFLOW_MODEL_ID)
    
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
            
            if confidence >= FISHING_CONFIDENCE_THRESHOLD:
                x1 = x - width / 2
                y1 = y - height / 2
                x2 = x + width / 2
                y2 = y + height / 2
                pred_boxes.append([x1, y1, x2, y2])
                pred_conf.append(confidence)
    
    boat_count = len(pred_boxes)
    
    return {
        'pred_classes': [],  # Not used for fishing model
        'pred_boxes': pred_boxes,
        'pred_conf': pred_conf,
        'in_count': None,
        'out_count': None,
        'boat_count': boat_count,
        'class_names': FISHING_CLASS_NAMES
    }
