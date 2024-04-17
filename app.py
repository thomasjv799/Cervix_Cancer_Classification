import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
import subprocess
import os
from tensorflow.keras.models import load_model
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# Command to run git clone of yolo repo
git_clone_command = ["git", "clone", "https://github.com/ultralytics/yolov5"]
current_path = os.getcwd()
mapping_dict = {0:'Stage 1',1:'Stage 2', 2: 'Stage 3'}


try:
    if not os.path.exists('yolov5'):
        subprocess.run(git_clone_command, check=True)
        print("Git clone successful.")
        os.chdir('yolov5')
        print("Directory Changed.")
except subprocess.CalledProcessError as e:
    print("Error during git clone:", e)


yolo_model_path = current_path + os.sep + "exp2/weights/best.pt"


# Function to perform object detection using YOLO
def yolo_object_detection(image, model_path=yolo_model_path):
   model = torch.hub.load(current_path + '/' + 'yolov5', 'custom', path=model_path, source='local')
   results = model(image)
   image_array_bgr = results.crop()[0]['im']
   image_array_rgb = cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2RGB)
   return image_array_bgr, image_array_rgb

def model_prediction(image):
    model = load_model(current_path + os.sep + 'densenet_cvx.h5')
    img = cv2.resize(image, (224,224),interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)
    img = img / 255
    img_exp = np.expand_dims(img,axis= 0)
    probs = model.predict(img_exp)[0]
    pred_label = np.argmax(probs)
    label = mapping_dict[pred_label]
    return label, probs[pred_label]
    
# Streamlit app layout
def main():
    st.title("Cervical Cancer Detection")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', width=350)

        # Perform object detection on button click
        if st.button('Predict'):
            with st.spinner('Detecting objects...'):
                # Perform YOLO object detection
                crop_image, rgb_image = yolo_object_detection(image)
                # Display the result
                st.image(rgb_image, caption='Region of Interest', width=350)
                label, proba =  model_prediction(crop_image)
                proba = proba*100
                proba = np.round(proba,2)
                st.write(f"The Model Predicts the label as {label} with {proba} probability.")

if __name__ == "__main__":
    main()
