
import subprocess
import sys
import os

# Ensure OpenCV is available
try:
    import cv2
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

import torch


from ultralytics import YOLO

model = YOLO("best.pt")


# Other imports
import streamlit as st
import numpy as np
import easyocr
from PIL import Image

# EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=".")

st.title("🚀 Helmet Detection & Number Plate Recognition")
st.write("Upload an image or video. The model detects helmet usage and reads number plates if helmet is missing.")

# Number plate text normalization
def is_valid_number_plate(text):
    dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
    dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}
    new_text = list(text)
    for i in range(len(new_text)):
        if new_text[i].isnumeric():
            new_text[i] = dict_int_to_char.get(new_text[i], new_text[i])
        elif new_text[i].isalpha():
            new_text[i] = dict_char_to_int.get(new_text[i], new_text[i])
    return ''.join(new_text) if 8 <= len(text) <= 10 else None

# File Upload
uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    if uploaded_file.type.startswith("image"):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image")
        img = np.array(img)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = model.predict(source=frame)
        number_plate_region = None

        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, _, class_id = map(int, box[:6])
            label = results[0].names.get(class_id, str(class_id))
            if label == "number plate":
                number_plate_region = (x1, y1, x2, y2)
            color = (0, 255, 0) if label == "with helmet" else (0, 0, 255) if label == "without helmet" else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="YOLO Prediction")

        if number_plate_region:
            x1, y1, x2, y2 = number_plate_region
            cropped = frame[y1:y2, x1:x2]
            st.image(cropped, caption="Number Plate Detected")
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            result = reader.readtext(gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                     decoder='greedy', detail=1, paragraph=False, adjust_contrast=0.8)

            detected_text = " ".join([d[1] for d in result]) if result else ""
            final_answer = detected_text.replace(" ", "")
            corrected_text = is_valid_number_plate(final_answer)
            confidence = result[0][2] if result and len(result[0]) > 2 else 0.0

            if corrected_text:
                st.success(f"Detected: {corrected_text} (Original: {final_answer}, Confidence: {confidence:.2f})")
            elif final_answer:
                st.warning(f"Detected: {final_answer} (Confidence: {confidence:.2f}) - Invalid Format")
            else:
                st.info("No text found on plate.")
        else:
            st.info("No number plate detected.")

    elif uploaded_file.type.startswith("video"):
        st.subheader("📹 Processing Video...")
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error("Failed to read video file.")
        else:
            frame_placeholder = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame)

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = model.names.get(int(box.cls[0]), str(box.cls[0]))
                        color = (0, 255, 0) if label == "with helmet" else (0, 0, 255) if label == "without helmet" else (255, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            cap.release()
            st.video(temp_path)
            os.remove(temp_path)
