
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
st.write("This app takes an image or video of motorcyclists and detects helmet usage.")
st.write("If helmet is not worn, it detects and extracts the number plate text.")

def is_valid_number_plate(text):
    dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
    dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

    new_text = list(text)
    for i, c in enumerate(new_text):
        if c.isnumeric():
            new_text[i] = dict_int_to_char.get(c, c)
        elif c.isalpha():
            new_text[i] = dict_char_to_int.get(c, c)
    return ''.join(new_text) if len(new_text) in [8, 9, 10] else None

uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    if uploaded_file.type.startswith("image"):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Input Image", use_container_width=True)
        img = np.array(img)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = model.predict(source=frame)
        number_plate_region = None

        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = map(int, box[:6])
            label = results[0].names[class_id]
            color = (0, 255, 0) if label == "with helmet" else (0, 0, 255) if label == "without helmet" else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if label == "number plate":
                number_plate_region = (x1, y1, x2, y2)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)

        if number_plate_region:
            x1, y1, x2, y2 = number_plate_region
            cropped = frame[y1:y2, x1:x2]
            st.image(cropped, caption="Number Plate Region Detected", use_container_width=True)
            gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            result = reader.readtext(gray_cropped, decoder='greedy', allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=1, adjust_contrast=0.8, paragraph=False, min_size=4)
            detected_text = " ".join([d[1] for d in result])
            final_answer = detected_text.replace(" ", "")
            corrected_text = is_valid_number_plate(final_answer)

            if corrected_text:
                st.write(f"**Detected Text:** {corrected_text} (Original: {final_answer}, Confidence: {result[0][2]:.2f})")
            else:
                st.write(f"**Detected Text:** {final_answer} (Confidence: {result[0][2]:.2f}) - Invalid Number Plate Format")

    elif uploaded_file.type.startswith("video"):
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        cap = cv2.VideoCapture(temp_file_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = model.predict(source=frame)
            number_plate_region = None

            for box in results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = map(int, box[:6])
                label = results[0].names[class_id]
                color = (0, 255, 0) if label == "with helmet" else (0, 0, 255) if label == "without helmet" else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if label == "number plate":
                    number_plate_region = (x1, y1, x2, y2)

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_count}", use_container_width=True)

            if number_plate_region:
                x1, y1, x2, y2 = number_plate_region
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    st.image(cropped, caption="Number Plate Region", use_container_width=True)
                    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    result = reader.readtext(gray_cropped, decoder='greedy', allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=1, adjust_contrast=0.8, paragraph=False, min_size=4)
                    detected_text = " ".join([d[1] for d in result]) if result else ""
                    final_answer = detected_text.replace(" ", "")
                    corrected_text = is_valid_number_plate(final_answer)
                    confidence_score = result[0][2] if result and len(result[0]) > 2 else 0.0

                    if corrected_text:
                        st.write(f"**Detected Text:** {corrected_text} (Original: {final_answer}, Confidence: {confidence_score:.2f})")
                    else:
                        st.write(f"**Detected Text:** {final_answer} (Confidence: {confidence_score:.2f}) - Invalid Format")

        cap.release()
        os.remove(temp_file_path)
