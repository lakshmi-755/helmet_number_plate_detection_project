
import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image
import imutils  # Import imutils

# Load YOLOv8 Model
model = YOLO(r"C:\Users\HP\runs\detect\train13\weights\best.pt")

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=True, model_storage_directory=r"C:\Users\HP\Downloads\project_dataset")


st.title("🚀 Helmet Detection & Number Plate Recognition")
st.write("This the web page in which it takes image or video of motorcyclists as input")
st.write("And detects whether they are wearing helmet or not")
st.write("If they they are not wearning helmet then their numberplate is detected ")
st.write("Model tries to extract the text on the number plate such that further actions can be taken")


def is_valid_number_plate(text):
    dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
    dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

    # Ensure the text length is exactly 8 (for the format XXNNXXNN)
    if len(text) == 8:
        new_text = list(text)  # Convert to a list to modify the characters
        if new_text[0].isnumeric():
            new_text[0] = dict_int_to_char.get(new_text[0], new_text[0])  # Use get to avoid KeyError
        if new_text[1].isnumeric():
            new_text[1] = dict_int_to_char.get(new_text[1], new_text[1])
        if new_text[2].isalpha():
            new_text[2] = dict_char_to_int.get(new_text[2], new_text[2])
        if new_text[3].isalpha():
            new_text[3] = dict_char_to_int.get(new_text[3], new_text[3])
        if new_text[4].isnumeric():
            new_text[4] = dict_int_to_char.get(new_text[4], new_text[4])
        if new_text[5].isalpha():
            new_text[5] = dict_char_to_int.get(new_text[5], new_text[5])
        if new_text[6].isalpha():
            new_text[6] = dict_char_to_int.get(new_text[6], new_text[6])
        if new_text[7].isalpha():
            new_text[7] = dict_char_to_int.get(new_text[7], new_text[7])
        return ''.join(new_text) 
    if len(text) == 9:
        new_text = list(text)  # Convert to a list to modify the characters
        if new_text[0].isnumeric():
            new_text[0] = dict_int_to_char.get(new_text[0], new_text[0])  # Use get to avoid KeyError
        if new_text[1].isnumeric():
            new_text[1] = dict_int_to_char.get(new_text[1], new_text[1])
        if new_text[2].isalpha():
            new_text[2] = dict_char_to_int.get(new_text[2], new_text[2])
        if new_text[3].isalpha():
            new_text[3] = dict_char_to_int.get(new_text[3], new_text[3])
        if new_text[4].isnumeric():
            new_text[4] = dict_int_to_char.get(new_text[4], new_text[4])
        if new_text[5].isalpha():
            new_text[5] = dict_char_to_int.get(new_text[5], new_text[5])
        if new_text[6].isalpha():
            new_text[6] = dict_char_to_int.get(new_text[6], new_text[6])
        if new_text[7].isalpha():
            new_text[7] = dict_char_to_int.get(new_text[7], new_text[7])
        if new_text[8].isalpha():
            new_text[8] = dict_char_to_int.get(new_text[8], new_text[8])
        return ''.join(new_text)

    if len(text) == 10:
        new_text = list(text)  # Convert to a list to modify the characters
        if new_text[0].isnumeric():
            new_text[0] = dict_int_to_char.get(new_text[0], new_text[0])  # Use get to avoid KeyError
        if new_text[1].isnumeric():
            new_text[1] = dict_int_to_char.get(new_text[1], new_text[1])
        if new_text[2].isalpha():
            new_text[2] = dict_char_to_int.get(new_text[2], new_text[2])
        if new_text[3].isalpha():
            new_text[3] = dict_char_to_int.get(new_text[3], new_text[3])
        if new_text[4].isnumeric():
            new_text[4] = dict_int_to_char.get(new_text[4], new_text[4])
        if new_text[5].isnumeric():
            new_text[5] = dict_int_to_char.get(new_text[5], new_text[5])
        if new_text[6].isalpha():
            new_text[6] = dict_char_to_int.get(new_text[6], new_text[6])
        if new_text[7].isalpha():
            new_text[7] = dict_char_to_int.get(new_text[7], new_text[7])
        if new_text[8].isalpha():
            new_text[8] = dict_char_to_int.get(new_text[8], new_text[8])
        if new_text[9].isalpha():
            new_text[9] = dict_char_to_int.get(new_text[9], new_text[9])
        return ''.join(new_text)

    return None

# File Upload
uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    if uploaded_file.type.startswith("image"):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img,caption="your input image is..")
        img = np.array(img)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV processing

        results = model.predict(source=frame)  # Use BGR frame for YOLO
        
        
        number_plate_region = None
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = map(int, box[:6])
            label = results[0].names[class_id]

            if label == "number plate":
                number_plate_region = (x1, y1, x2, y2)

            color = (0, 255, 0) if label == "with helmet" else (0, 0, 255) if label == "without helmet" else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # Use line_thickness = 2
            cv2.putText(frame, f"{label}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) # Use font_thickness = 1
        st.write("The labeled image by the yolov8n model is..")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True) # Convert back to RGB for Streamlit

        if number_plate_region:
            x1, y1, x2, y2 = number_plate_region
            cropped = frame[y1:y2, x1:x2]  # Crop using BGR frame
            st.image(cropped, caption=" Number Plate Region Detected is", use_column_width=True)
            gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            result = reader.readtext(gray_cropped,
                                     decoder='greedy',
                                     allowlist='A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9',
                                     detail=1,
                                     adjust_contrast=0.8,
                                     paragraph=False,
                                     min_size=4)

            detected_text = " ".join([detection[1] for detection in result])
            final_answer = detected_text.replace(" ", "")
            corrected_text = is_valid_number_plate(final_answer)

            if corrected_text:
                st.write(f"**Detected Text:** {corrected_text} (Original: {final_answer}, Confidence: {result[0][2]:.2f} ")  # Display corrected text
            else:
                st.write(f"**Detected Text:** {final_answer} (Confidence: {result[0][2]:.2f} - Invalid Number Plate Format")  # Display original if correction fails

             

        else:
             st.write("Number plate region not detected.")


    elif uploaded_file.type.startswith("video"):
        st.subheader("📹 Video Processing...")
        temp_file_path = f"temp_{uploaded_file.name}"

        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        cap = cv2.VideoCapture(temp_file_path)

        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            frame_count = 0
            number_plate_region = None
            frame_placeholder = st.empty()  # Placeholder for real-time frame update

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_count += 1
                results = model.predict(source=frame)

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = model.names[int(box.cls[0])]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for box in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, confidence, class_id = map(int, box[:6])
                    label = results[0].names[class_id]

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
                        st.image(cropped, caption="Number Plate Region Detected", use_column_width=True)
                        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

                        result = reader.readtext(
                            gray_cropped,
                            decoder='greedy',
                            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                            detail=1,
                            adjust_contrast=0.8,
                            paragraph=False,
                            min_size=4
                        )

                        detected_text = " ".join([detection[1] for detection in result]) if result else ""
                        final_answer = detected_text.replace(" ", "")
                        corrected_text = is_valid_number_plate(final_answer)

                        confidence_score = result[0][2] if result and len(result[0]) > 2 else 0.0  # Avoid IndexError

                        if corrected_text:
                            st.write(f"**Detected Text:** {corrected_text} (Original: {final_answer}, Confidence: {confidence_score:.2f})")
                        else:
                            st.write(f"**Detected Text:** {final_answer} (Confidence: {confidence_score:.2f}) - Invalid Number Plate Format")
                    else:
                        st.write("Error: Could not extract number plate region.")

            cap.release()

            # Show final processed video
            st.video(temp_file_path)

            # Clean up temporary file
            os.remove(temp_file_path)
