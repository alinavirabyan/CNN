# Object Detection with YOLOv8 on Pascal VOC 2012
## Object Detection and YOLO
Object detection is a computer vision technique used to identify and localize objects within an image. YOLO (You Only Look Once) is a real-time object detection model known for its speed and accuracy, capable of detecting multiple objects in a single forward pass of the network.

### About This Project

This project applies the **YOLOv8** model to the **Pascal VOC 2012** dataset to detect 20 different object classes such as people, vehicles, and animals. The dataset is preprocessed and converted into YOLO format, and a YOLOv8n model is trained using Ultralytics' implementation. The model is evaluated using precision, recall, and mAP metrics and then used for inference on sample images.

## Full code in Jupyter Notebook:   
[ 3.ipynb on GitHub](https://github.com/alinavirabyan/CNN/blob/main/3.ipynb)

### Project Structure



### What This Code Does
This project implements a full object detection pipeline using YOLOv8 on the Pascal VOC 2012 dataset. The process is fully handled in the notebook, organized into the following key steps:

#### 1. Environment Setup
Installs required libraries such as YOLOv8 (ultralytics), torchvision, albumentations, matplotlib, and more.

#### 2. Configuration
Defines a configuration class containing:
Dataset paths and structure
Model type (yolov8n.pt)
The 20 Pascal VOC object classes
Training hyperparameters (batch size, learning rate, epochs, etc.)

####  3. Dataset Preparation
Downloads the Pascal VOC 2012 dataset using torchvision.
Creates a directory structure compatible with YOLO training (images and labels in separate folders for training and validation).

####  4. Annotation Conversion
Converts XML annotations from Pascal VOC format to YOLO .txt format.
Normalizes bounding box coordinates and matches each label to its corresponding image.

#### 5. Data Processing
Processes a limited number of images (e.g., 500 for training, 100 for validation) for faster training.
Copies images and generates YOLO-compatible label files.

#### 6. YOLO Dataset Configuration
Generates a voc.yaml file, which tells the YOLO model:
Where to find training and validation data
The names and indices of object classes

#### 7. Model Training
Loads and trains the YOLOv8 model using the pre-defined configuration.
Automatically detects and uses a GPU if available.
Saves the trained model and logs metrics (e.g., loss, precision, recall, mAP).

#### 8. Evaluation & Visualization
Plots training loss and evaluation metrics (box loss, classification loss, precision, recall, mAP@0.5) over time using graphs.
Displays final performance metrics: These numbers represent the model’s performance: Precision (0.7144) shows how accurate the detections are, Recall (0.6255) indicates how many actual objects were correctly detected, and mAP@0.5 (0.6550) reflects the overall detection quality at an IoU threshold of 0.5
Precision: 0.7144
Recall: 0.6255
mAP@0.5: 0.6550

#### 9. Inference
Loads the best model checkpoint.
Allows user to upload an image and performs object detection on it.
Visualizes the detection results with bounding boxes and class labels.

![image](https://github.com/user-attachments/assets/fc75dfc8-3200-4799-9050-0439773de1ed)
The "Losses over Epochs" graph shows how the model's errors decrease as it trains over time. Lower box and classification losses indicate that the model is improving in both locating and correctly identifying objects.

![image](https://github.com/user-attachments/assets/d12b281a-0e65-4350-b18b-b921515bda4c)
The "Precision, Recall, and mAP over Epochs" graph shows how the model's detection performance improves throughout training. Increasing values indicate better accuracy in identifying objects (precision), finding all relevant objects (recall), and overall detection quality (mAP).

This model is trained to detect 20 object classes from the Pascal VOC dataset, including **person, car, dog, cat, bicycle, airplane, chair, and more**. These classes are defined in the code and used throughout the training and evaluation process.

| Notebook                  | Image Size | Batch Size | Epochs        | Precision  | Recall     | mAP\@0.5   | Link                                                             |
| ------------------------- | ---------- | ---------- | ------------- | ---------- | ---------- | ---------- | ---------------------------------------------------------------- |
| `3.ipynb` *(Best Result)* | 640        | 16         | 200           | **0.7144** | **0.6255** | **0.6550** | [View](https://github.com/alinavirabyan/CNN/blob/main/3.ipynb)   |
| `3_2.ipynb`               | 1024       | 16         | (unspecified) | 0.6841     | 0.4868     | 0.5912     | [View](https://github.com/alinavirabyan/CNN/blob/main/3_2.ipynb) |
| `3_1.ipynb`               | 640        | 32         | 150           | 0.6404     | 0.5944     | 0.6260     | [View](https://github.com/alinavirabyan/CNN/blob/main/3_1.ipynb) |
























