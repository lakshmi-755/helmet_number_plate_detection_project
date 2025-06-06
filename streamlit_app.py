
import subprocess
import sys
import os

# Ensure OpenCV is available
try:
    import cv2
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="torchscript")

# Fix for PyTorch >= 2.6 model unpickling
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
import ultralytics.nn.modules.conv

add_safe_globals([
    DetectionModel,
    Sequential,
    ultralytics.nn.modules.conv.Conv
])

import streamlit as st
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image
import imutils

# Load YOLOv8 Model
model = YOLO("best.torchscript.pt")

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=".")

st.title("🚀 Helmet Detection & Number Plate Recognition")
st.write("This app accepts an image or video of motorcyclists as input.")
st.write("It detects whether they are wearing a helmet.")
st.write("If not, it extracts and reads the number plate.")

# Number plate cleanup logic
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

# File upload UI
uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    if uploaded_file.type.startswith("image"):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Your input image is...")
        img = np.array(img)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = model.predict(source=frame)
        number_plate_region = None

        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = map(int, box[:6])
            label = results[0].names.get(class_id, str(class_id))
            if label == "number plate":
                number_plate_region = (x1, y1, x2, y2)
            color = (0, 255, 0) if label == "with helmet" else (0, 0, 255) if label == "without helmet" else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        st.write("🖼️ YOLO Model Prediction Output:")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

        if number_plate_region:
            x1, y1, x2, y2 = number_plate_region
            cropped = frame[y1:y2, x1:x2]
            st.image(cropped, caption="Number Plate Region Detected", use_column_width=True)
            gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            result = reader.readtext(gray_cropped,
                                     decoder='greedy',
                                     allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                     detail=1,
                                     adjust_contrast=0.8,
                                     paragraph=False,
                                     min_size=4)

            detected_text = " ".join([detection[1] for detection in result]) if result else ""
            final_answer = detected_text.replace(" ", "")
            corrected_text = is_valid_number_plate(final_answer)
            confidence_score = result[0][2] if result and len(result[0]) > 2 else 0.0

            if corrected_text:
                st.write(f"**Detected Text:** {corrected_text} (Original: {final_answer}, Confidence: {confidence_score:.2f})")
            elif final_answer:
                st.write(f"**Detected Text:** {final_answer} (Confidence: {confidence_score:.2f}) - Invalid Number Plate Format")
            else:
                st.write("🔍 No recognizable text found on the number plate.")
        else:
            st.write("🚫 Number plate region not detected.")

    elif uploaded_file.type.startswith("video"):
        st.subheader("📹 Processing video...")
        temp_file_path = f"temp_{uploaded_file.name}"

        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        cap = cv2.VideoCapture(temp_file_path)

        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            frame_count = 0
            number_plate_region = None
            frame_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_count += 1
                results = model.predict(source=frame)

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = model.names.get(int(box.cls[0]), str(box.cls[0]))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for box in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, confidence, class_id = map(int, box[:6])
                    label = results[0].names.get(class_id, str(class_id))
                    if label == "number plate":
                        number_plate_region = (x1, y1, x2, y2)

                    color = (0, 255, 0) if label == "with helmet" else (0, 0, 255) if label == "without helmet" else (255, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Processing Frame {frame_count}", use_column_width=True)

                if number_plate_region:
                    x1, y1, x2, y2 = number_plate_region
                    cropped = frame[y1:y2, x1:x2]
                    if cropped is not None and cropped.size != 0:
                        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                        result = reader.readtext(gray_cropped,
                                                 decoder='greedy',
                                                 allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                                 detail=1,
                                                 adjust_contrast=0.8,
                                                 paragraph=False,
                                                 min_size=4)

                        detected_text = " ".join([detection[1] for detection in result]) if result else ""
                        final_answer = detected_text.replace(" ", "")
                        corrected_text = is_valid_number_plate(final_answer)
                        confidence_score = result[0][2] if result and len(result[0]) > 2 else 0.0

                        if corrected_text:
                            st.write(f"**Detected Text:** {corrected_text} (Original: {final_answer}, Confidence: {confidence_score:.2f})")
                        elif final_answer:
                            st.write(f"**Detected Text:** {final_answer} (Confidence: {confidence_score:.2f}) - Invalid Number Plate Format")

        cap.release()
        st.video(temp_file_path)
        os.remove(temp_file_path)
