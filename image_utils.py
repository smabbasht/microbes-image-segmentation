import os
from PIL import Image
import streamlit as st
from streamlit_image_comparison import image_comparison
import cv2
def convert_to_jpg(uploaded_image):
    im = Image.open(uploaded_image)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    uploaded_image_path = os.path.join("Images", "uploaded_image.jpg")
    im.save(uploaded_image_path)
    return uploaded_image_path

def detect_microbes(model, img_file):
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
    
    return names_list
