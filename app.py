
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
st.title("Fire image Detection App :)")

# Load YOLO model
# model = YOLO("runs/detect/train73/weights/best.pt")
model = YOLO("best_model2.pt")
# Upload image
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
  # Show original image
  st.image(uploaded_image , caption="Uploaded Image", use_container_width=True)

  # Read image and convert to numpy array
  image = Image.open(uploaded_image)
  image_np = np.array(image)

  # Run YOLO inference
  st.info("Running YOLO object detection...")
  results = model.predict(image_np , conf=0.4)

  # Draw results on image
  result_image = results[0].plot()
  st.image(result_image , caption="YOLO Detection Result", use_container_width=
  True)
  st.success("Detection completed!")

  # Extract detection results
  boxes = results[0].boxes
  class_ids = boxes.cls.cpu().numpy().astype(int)
  class_names = [model.names[i] for i in class_ids]

  # Count Fire
  fire_count = class_names.count("fire")
  st.write(f"Number of fire detected: **{fire_count}**")
