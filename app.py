import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
import subprocess
import os
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# Command to run git clone
git_clone_command = ["git", "clone", "https://github.com/ultralytics/yolov5"]
current_path = os.getcwd()

try:
    if not os.path.exists('yolov5'):
        subprocess.run(git_clone_command, check=True)
        print("Git clone successful.")
        os.chdir('yolov5')
        print("Directory Changed.")
except subprocess.CalledProcessError as e:
    print("Error during git clone:", e)


# Function to perform object detection using YOLO
def yolo_object_detection(image, model_path="D:/CVX/Cervix_Cancer_Classification/exp2/weights/best.pt"):
   model = torch.hub.load(current_path + '/' + 'yolov5', 'custom', path=model_path, source='local')
   results = model(image)
   image_array_bgr = results.crop()[0]['im']
   image_array_rgb = cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2RGB)
   return image_array_rgb

# Streamlit app layout
def main():
    st.title("YOLO Object Detection App")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', width=350)

        # Perform object detection on button click
        if st.button('Predict'):
            with st.spinner('Detecting objects...'):
                # Perform YOLO object detection
                detected_image = yolo_object_detection(image)
                # Display the result
                st.image(detected_image, caption='Objects Detected', width=350)

if __name__ == "__main__":
    main()
