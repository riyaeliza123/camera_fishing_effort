# Documentation

This document contains definitions, expectations and hypothesis.

## Goal of this project

The objective is to detect the existance of relevant fishing boats in the images:
- For chokepoint camera model - the goal is to count the number of boats coming in v/s going out
- For fishing point model - the goal is to count the number of boats fishing (stationary)

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

Here, we **do not** care about exact bounding box quality and tight localization. Since there is always a trade-off between recall and precision, in this case recall>precision. 

Recall here is more important because a missed boat = wrong decision.

Expected metrics for this model are the following:

- Recall >= 0.7
- Precision >=0.3
- mAP50 >= 0.5
- mAP50-95 >= 0.15