# Documentation

This document contains goals, definitions and expectations.

## Goal of this project

The objective is to detect the existance of relevant fishing boats in the images:
- For chokepoint camera model - the goal is to count the number of boats coming in v/s going out
- For fishing point model - the goal is to count the number of boats fishing (stationary)

## Definitions and Concepts

### YOLO (You Only Look Once)
YOLO is an object detection technology that identifies and locates objects in images in real-time. Instead of scanning an image multiple times, YOLO analyzes the entire image at once and predicts where objects are and what they are. In this project, YOLO is trained to recognize fishing boats and classify them as either "in" (entering) or "out" (leaving) the monitoring area.

### Model Training
Model training is the process of teaching a YOLO model to recognize specific objects by showing it many labeled examples. In our project:
- We provide hundreds of annotated fishing boat images where each boat is marked with a box and labeled
- The YOLO model learns patterns from these examples
- The trained model can then identify similar boats in new, unseen images
- Training involves adjusting internal parameters iteratively until the model achieves good accuracy

### Web Technologies
Our application uses the following web technologies:
- **FastAPI**: A modern Python framework for building web applications that receive image uploads and send back analysis results
- **Jinja2**: A templating engine that generates HTML pages users see in their browser
- **HTML/CSS/JavaScript**: Frontend technologies that create the user interface for uploading images and viewing results

### Dockerization
Docker is a containerization technology that packages an application with all its dependencies (Python, libraries, model weights) into a single unit called a "container". This ensures the application runs consistently across different computers. We use Docker to:
- Package the entire camera fishing effort application
- Guarantee it works the same way on local machines, cloud servers, and deployment platforms
- Make deployment and scaling easier

### Fly.io
Fly.io is a cloud platform that hosts and runs Docker containers. Instead of running the application on your personal computer, Fly.io runs it on their servers, making it accessible from anywhere on the internet. The application stays running 24/7 and can handle multiple users simultaneously.

### Roboflow Dataset Annotation
Roboflow is a platform for managing training datasets. Dataset annotation is the process of:
- Uploading raw images to Roboflow
- Drawing bounding boxes around objects of interest (fishing boats)
- Labeling each box with its class (e.g., "in" or "out")
- Exporting the annotated dataset in a format YOLO can use for training
Proper annotation is crucial because the quality of training data directly affects model accuracy.

### Confidence Threshold
In our predictions, a confidence score (0-1) indicates how certain the model is about a detection. A confidence threshold of 0.5 means we only count detections where the model is at least 50% confident. This filters out uncertain predictions and improves result reliability.

## Metrics of YOLO object detection models

Accuracy is not a suitable metric for image models since accuracy assumes there is only one prediction per image. In object detection, an image may contain multiple objects, multiple predictions and large background regions. Instead YOLO provides thefollowing metrics:

1. Recall : Measures how many of the actual objects in the image are successfully detected by the model.
2. Precision : Measures how many objects detected by the model are correct.
3. mAP50 :  Measures the model’s overall detection performance by averaging precision across all classes, where a detection is considered correct if the Intersection over Union (IoU) with the ground-truth bounding box is at least 50%. It evaluates whether the model can find objects reasonably well.
4. mAP50-95 : The average of mAP values computed at multiple IoU thresholds, from 0.50 to 0.95 in steps of 0.05. This metric measures both detection and localization accuracy, penalizing poorly aligned bounding boxes. It is considered the most strict and comprehensive YOLO metric.

## Definition of success
### 1. Chokepoint model
Task: Binary existence detection 
- “Is there at least one `in` boat? If so, how many?”
- “Is there at least one `out` boat? If so, how many?”

For this model, the exact position and size of the bounding box around a boat is less critical than simply detecting whether boats are present. A loosely drawn box is acceptable as long as the boat is identified.

There is always a trade-off between recall and precision. We prioritize recall (catching all boats) over precision (perfect predictions) because:
- **Missed boats have serious consequences**: If a boat enters or exits the monitoring area but our model fails to detect it, the count will be wrong and the fishing effort data will be inaccurate
- **False positives are more tolerable**: If the model occasionally detects something that isn't a boat, it has less impact—the false count can be manually reviewed or filtered. The worst outcome is a slightly inflated count

Expected metrics for this model are the following:

- Recall >= 0.7
- Precision >= 0.3
- mAP50 >= 0.5

### 2. Fishing point model
Task: Generic boat detection
- "Is there at least one fishing boat in the image? If so, how many?"

This model focuses on simple detection of any fishing boat without classification into categories. The goal is to identify the presence and count of boats engaged in fishing activity (stationary or active fishing).

Here, recall is equally important as precision since both missed boats and false positives affect the fishing effort count. A balanced approach is necessary for reliable boat counting.

Expected metrics for this model are the following:

- Recall >= 0.65
- Precision >= 0.65
- mAP50 >= 0.6

