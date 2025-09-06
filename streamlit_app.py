
import os
import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image

# -----------------------------
# Load YOLOv8 Model
# -----------------------------
model = YOLO("best.pt")

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=".")

st.title("🚀 Helmet Detection & Number Plate Recognition")
st.write("Upload an image or video to detect helmets and recognize number plates.")

# -----------------------------
# Helper Function: Validate and correct number plate
# -----------------------------
def is_valid_number_plate(text):
    dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
    dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}
    text = text.upper().replace(" ", "")
    new_text = list(text)
    for i in range(len(new_text)):
        if new_text[i].isnumeric():
            new_text[i] = dict_int_to_char.get(new_text[i], new_text[i])
        else:
            new_text[i] = dict_char_to_int.get(new_text[i], new_text[i])
    return ''.join(new_text) if 7 <= len(text) <= 10 else None

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    # -----------------------------
    # IMAGE PROCESSING
    # -----------------------------
    if uploaded_file.type.startswith("image"):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Input Image")
        img = np.array(img)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = model.predict(source=frame)
        number_plate_region = None

        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = map(int, box[:6])
            label = results[0].names[class_id]

            if label == "number plate":
                number_plate_region = (x1, y1, x2, y2)

            # Draw bounding box
            color = (0, 255, 0) if label == "with helmet" else (0, 0, 255) if label == "without helmet" else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detected Output", use_container_width=True)

        # OCR for number plate
        if number_plate_region:
            x1, y1, x2, y2 = number_plate_region
            cropped = frame[y1:y2, x1:x2]
            st.image(cropped, caption="Detected Number Plate", use_container_width=True)
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            result = reader.readtext(gray, decoder='greedy',
                                     allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=1)

            detected_text = " ".join([r[1] for r in result])
            final_text = detected_text.replace(" ", "")
            corrected = is_valid_number_plate(final_text)

            if corrected:
                st.write(f"**Detected Text:** {corrected} (Original: {final_text})")
            else:
                st.write(f"**Detected Text:** {final_text} - Invalid Format")

    # -----------------------------
    # VIDEO PROCESSING
    # -----------------------------
    elif uploaded_file.type.startswith("video"):
        st.subheader("📹 Video Processing...")

        # Save uploaded video temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        # Show original uploaded video
        st.video(temp_file_path)

        # Define output path for processed video
        output_path = f"processed_{uploaded_file.name}"

        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # VideoWriter to save processed video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame)
                number_plate_region = None

                for box in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = map(int, box[:6])
                    label = results[0].names[class_id]

                    if label == "number plate":
                        number_plate_region = (x1, y1, x2, y2)

                    # Draw bounding boxes
                    color = (0, 255, 0) if label == "with helmet" else (0, 0, 255) if label == "without helmet" else (255, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Write processed frame to output video
                out.write(frame)

            cap.release()
            out.release()

            # Show processed video
            st.success("✅ Processed video with labels is ready!")
            st.video(output_path)

            # Clean up
            os.remove(temp_file_path)

