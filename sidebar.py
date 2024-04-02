import os
import streamlit as st
import pandas as pd
from image_utils import detect_microbes, convert_to_jpg

def sidebar_content(model):
    st.sidebar.write(
    """
This is a computer-aided application designed to segment your input images using the powerful YOLOv8 object detection algorithm developed by *Ultralytics*.

Just upload your image, and it will be segmented in real-time.

    """
)
    st.sidebar.divider()

    uploaded_file = st.sidebar.file_uploader("Drop a JPG/PNG file", accept_multiple_files=False, type=['jpg', 'png'])
    if uploaded_file is not None:
        uploaded_image_path = convert_to_jpg(uploaded_file)
        st.sidebar.success("File saved successfully")

    test_images = {
        "Image 1": "/home/hussain/img-segmentation/Images/check1.png",
        "Image 2": "/home/hussain/img-segmentation/Images/check2.png",
        "Image 3": "/home/hussain/img-segmentation/Images/check3.png",
        "Image 4": "/home/hussain/img-segmentation/Images/check.png"
    }
    selected_test_image = st.sidebar.selectbox("Select Test Image", list(test_images.keys()))
    if selected_test_image:
        img_file = os.path.join('Images', test_images[selected_test_image])

    names_list = detect_microbes(model, img_file)

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
