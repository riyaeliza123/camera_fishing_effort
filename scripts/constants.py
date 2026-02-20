"""Constants for the application."""

# Model paths and URLs
MODEL_PATH = "best.pt"
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/riyaeliza123/camera_fishing_effort/main/notebooks/runs/detect/chokepoint_finetuned/train/weights/best.pt"

# Confidence thresholds
CHOKEPOINT_CONFIDENCE_THRESHOLD = 0.5
FISHING_CONFIDENCE_THRESHOLD = 0.7

# Class names
CHOKEPOINT_CLASS_NAMES = ['in', 'out']
FISHING_CLASS_NAMES = ['boat']

# Roboflow model ID
ROBOFLOW_MODEL_ID = "fishing-access-points/2"
ROBOFLOW_API_URL = "https://detect.roboflow.com"

# Image processing
DEFAULT_IMAGE_SIZE = 640
