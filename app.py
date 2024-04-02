import os
import cv2
import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from streamlit_image_comparison import image_comparison
from image_utils import convert_to_jpg, detect_microbes
from sidebar import sidebar_content

# Set title and subheader
st.title("Real-Time Microbe Detection with YOLOv8: A Web Integration")
st.subheader("Implementing YOLOv8 for Microbe Detection")

# Write description
st.write("Microbe detection is crucial in various fields, especially in environmental monitoring and public health. YOLOv8, a cutting-edge deep learning model, offers high accuracy and efficiency in object detection tasks, including microorganism identification. In this web application, we showcase the utilization of YOLOv8 for real-time microbe detection. Users can upload images containing microbial cells and observe the segmented area, providing precise locations of the detected microbes. This interactive platform aims to enhance understanding and practical implementation of YOLOv8 in microbe detection tasks.")

# Add link to GitHub repository
url = "https://github.com/SYED-M-HUSSAIN/Microbial-cell-segmentation"
link = f'<a href="{url}">GitHub Repository here</a>'
st.markdown(link, unsafe_allow_html=True)

# Load YOLOv8 model
model = YOLO('best.pt')

# Add sidebar content
sidebar_content(model)
