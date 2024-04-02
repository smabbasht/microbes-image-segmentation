import os
import cv2
from PIL import Image
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from streamlit_image_comparison import image_comparison

model = YOLO('best.pt')

import streamlit as st

st.title("Real-Time Microbe Detection with YOLOv8: A Web Integration")
st.subheader("Implementing YOLOv8 for Microbe Detection")
st.write("Microbe detection is crucial in various fields, especially in environmental monitoring and public health. YOLOv8, a cutting-edge deep learning model, offers high accuracy and efficiency in object detection tasks, including microorganism identification. In this web application, we showcase the utilization of YOLOv8 for real-time microbe detection. Users can upload images containing microbial cells and observe the segmented area, providing precise locations of the detected microbes. This interactive platform aims to enhance understanding and practical implementation of YOLOv8 in microbe detection tasks.")


url = "https://github.com/SYED-M-HUSSAIN/Microbial-cell-segmentation"
link = f'<a href="{url}">GitHub Repository here</a>'
st.markdown(link, unsafe_allow_html=True)

def convert_to_jpg(uploaded_image):
    im = Image.open(uploaded_image)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    uploaded_image_path = os.path.join("Images", "check.png")
    im.save(uploaded_image_path)

st.divider()

st.markdown('')
st.markdown('##### Segmented Instances')

parent_media_path = "Images"

APPLICATION_MODE = "Upload Picture"

st.sidebar.write(
    """
This is a computer-aided application designed to segment your input images using the powerful YOLOv8 object detection algorithm developed by *Ultralytics*.

Just upload your image, and it will be segmented in real-time.

    """
)
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader("Drop a JPG/PNG file", accept_multiple_files=False, type=['jpg', 'png'])
if uploaded_file is not None:
    convert_to_jpg(uploaded_file)
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    new_file_name = "check.png"
    with open(os.path.join(parent_media_path, new_file_name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    img_file = os.path.join(parent_media_path, new_file_name)
    st.sidebar.success("File saved successfully")

test_images = {
    "Image 1": "/home/hussain/img-segmentation/Images/check1.png",
    "Image 2": "/home/hussain/img-segmentation/Images/check2.png",
    "Image 3": "/home/hussain/img-segmentation/Images/check3.png",
    "Image 4": "/home/hussain/img-segmentation/Images/check.png"
}
selected_test_image = st.sidebar.selectbox("Select Test Image", list(test_images.keys()))
if selected_test_image:
    img_file = os.path.join(parent_media_path, test_images[selected_test_image])

results = model(img_file)
img = cv2.imread(img_file)
names_list = []
for result in results:
    boxes = result.boxes.cpu().numpy()
    numCols = len(boxes)
    if numCols > 0:
        cols = st.columns(numCols)
    else:
        print(f"Number of Boxes found: {numCols}")
        st.warning("Unable to identify distinct items - Please retry with a clearer image")
    for box in boxes:
        r = box.xyxy[0].astype(int)
        rect = cv2.rectangle(img, r[:2], r[2:], (0, 0, 255), 2)


    st.markdown('')
    st.markdown('##### Slider of Uploaded Image and Segments')
    image_comparison(
        img1=img_file,
        img2=img,
        label1="Original Image",
        label2="Segmented Image",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )
    for i, box in enumerate(boxes):
        r = box.xyxy[0].astype(int)
        crop = img[r[1]:r[3], r[0]:r[2]]
        predicted_name = result.names[int(box.cls[0])]
        names_list.append(predicted_name)
        with cols[i]:
            st.write(str(predicted_name) + ".jpg")
            st.image(crop)

st.sidebar.divider()
st.sidebar.markdown('')
st.sidebar.markdown('#### Distribution of identified items')

st.sidebar.checkbox("Use container width", value=False, key="use_container_width")
if len(names_list) > 0:
    df_x = pd.DataFrame(names_list)
    summary_table = df_x[0].value_counts().rename_axis('unique_values').reset_index(name='counts')
    st.sidebar.dataframe(summary_table, use_container_width=st.session_state.use_container_width)
else:
    st.sidebar.warning("Unable to identify distinct items - Please retry with a clearer image")

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.sidebar.divider()
st.sidebar.info("Made with ❤ by the Syed M. Hussain")
