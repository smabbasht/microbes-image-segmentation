# Microbial Cell Detection with YOLOv8

This project implements a web application for real-time detection and segmentation of microbial cells using the YOLOv8 deep learning model.

## Overview

Microbial cell detection is essential in various fields, including environmental monitoring, public health, and biotechnology. This application provides a user-friendly interface for researchers and practitioners to quickly and accurately identify microbial cells in images.
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/SYED-M-HUSSAIN/Microbial-cell-segmentation/blob/main/readmefiles/img1.png" alt="Segmented Results 1" style="width: 45%; margin-right: 10px;">
    <img src="https://github.com/SYED-M-HUSSAIN/Microbial-cell-segmentation/blob/main/readmefiles/img2.png" alt="Segmented Results 2" style="width: 45%; margin-left: 10px;">
</div>

## Features

- Utilizes YOLOv8 for real-time object detection and segmentation.
- Web-based interface for easy interaction and visualization.
- Supports uploading of images containing microbial cells.
- Displays segmented regions and precise locations of detected cells.
- Provides distribution analysis of identified microbial cells.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/SYED-M-HUSSAIN/Microbial-cell-segmentation.git
```

2. Install the required dependencies:
   
```bash
  pip install -r requirements.txt
```
3. Run the main file:
   
```bash
  streamlit run app.py
```

## Directory Structure
```
.
├── README.md                # Project documentation
├── app.py                   # Main application script
├── best.pt                  # Pre-trained YOLOv8 model
├── image_utils.py           # Utility functions for image processing
├── segmentation.py          # Script for segmentation functionality
├── sidebar.py               # Script for sidebar components
├── Images                    # Directory to store uploaded images
│   └── uploaded_image.jpg   # Example uploaded image
├── .gitignore               # Git ignore file
└── requirements.txt         # Dependencies

```

## Website Link
```
https://microbial-cell-detection-yolov8.streamlit.app/

```


