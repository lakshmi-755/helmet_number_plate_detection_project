
import os
import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 Model
model = YOLO("best.pt")

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=".")

st.title("🚀 Helmet Detection & Number Plate Recognition")
st.write("Upload an image or video of motorcyclists to detect helmets and read number plates.")

def is_valid_number_plate(text):
    dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
    dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

    length = len(text)
    if length not in [8, 9, 10]:
        return None

    new_text = list(text)
    # Correct character mappings depending on position, numeric or alpha
    for i in range(length):
        c = new_text[i]
        if i in [0,1,4,5] and c.isnumeric():
            new_text[i] = dict_int_to_char.get(c, c)
        elif c.isalpha():
            new_text[i] = dict_char_to_int.get(c, c)
    return ''.join(new_text)

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

def display_cropped_plate(cropped_img):
    if cropped_img is None or cropped_img.size == 0:
        return
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    # Enhance contrast slightly
    gray = cv2.equalizeHist(gray)
    st.image(gray, caption="Number Plate Region Detected", width=400)

    result = reader.readtext(
        gray,
        decoder='greedy',
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1,
        paragraph=False,
        min_size=4
    )

    if not result:
        st.write("No text detected on number plate.")
        return

    detected_text = "".join([res[1] for res in result]).replace(" ", "")
    corrected_text = is_valid_number_plate(detected_text)
    confidence_score = result[0][2] if result and len(result[0]) > 2 else 0.0

    if corrected_text:
        st.write(f"**Detected Number Plate:** {corrected_text} (Original: {detected_text}, Confidence: {confidence_score:.2f})")
    else:
        st.write(f"**Detected Text:** {detected_text} (Confidence: {confidence_score:.2f}) - Invalid Number Plate Format")

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    if uploaded_file.type.startswith("image"):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Input Image", width=600)

        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = model.predict(source=frame)

        number_plate_region = None
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = box[:6]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = results[0].names[int(class_id)]

            color = (0, 255, 0) if label == "with helmet" else (0, 0, 255) if label == "without helmet" else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if label == "number plate":
                number_plate_region = (x1, y1, x2, y2)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Image", width=600)

        if number_plate_region:
            x1, y1, x2, y2 = number_plate_region
            cropped = frame[y1:y2, x1:x2]
            display_cropped_plate(cropped)
        else:
            st.write("No number plate detected.")

    elif uploaded_file.type.startswith("video"):
        st.subheader("📹 Video Processing...")
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error("Error opening video file.")
        else:
            frame_placeholder = st.empty()
            plate_placeholder = st.empty()
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                results = model.predict(source=frame)

                number_plate_region = None
                for box in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = box[:6]
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    label = results[0].names[int(class_id)]

                    color = (0, 255, 0) if label == "with helmet" else (0, 0, 255) if label == "without helmet" else (255, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if label == "number plate":
                        number_plate_region = (x1, y1, x2, y2)

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_count}", width=700)

                if number_plate_region:
                    x1, y1, x2, y2 = number_plate_region
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size != 0:
                        # Update plate placeholder separately to avoid flickering frame
                        with plate_placeholder.container():
                            display_cropped_plate(cropped)
                    else:
                        plate_placeholder.empty()
                else:
                    plate_placeholder.empty()

            cap.release()
            st.video(temp_path)
            os.remove(temp_path)
