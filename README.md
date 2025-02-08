# helmet_number_plate_detection_project
 Takes image or video of motorcyclists as input  And detects whether they are wearing helmet or not  If they they are not wearning helmet then their numberplate is detected  Model tries to extract the text on the number plate such that further actions can be taken
🚀 Helmet and Number Plate Recognition for Motorcycles

📌 Project Overview
This project utilizes Deep Learning to detect motorcyclists and recognize whether they are wearing helmets. If a rider is found without a helmet, the system identifies their number plate and extracts the text for further action.

🎯 Objective
Detect motorcycles and riders in images/videos.
Check whether the rider is wearing a helmet.
Identify riders without helmets and extract their vehicle's number plate.
Display the recognized number plate text for further processing.
🛠️ Tech Stack
Deep Learning Model: YOLOv8n (fine-tuned for helmet detection)
Libraries Used:
Streamlit - Web app deployment
OpenCV - Image processing
NumPy - Array operations
Pandas - Data handling
Matplotlib - Visualization
Imutils, PIL - Image handling
EasyOCR - Text recognition (for number plates)
📂 Dataset
We prepared a custom dataset by collecting publicly available images and annotating them using Roboflow. The dataset follows the YOLOv8 format:
Training Set - 70%
Validation Set - 20%
Testing Set - 10%
🔍 Methodology
1️⃣ Object Detection with YOLOv8n
YOLOv8n is pre-trained on the MS-COCO dataset.
Fine-tuned to detect motorcycles, riders, and helmets.
Custom labels:
🟢 Green → Rider with helmet
🔴 Red → Rider without helmet
2️⃣ Helmet Detection Logic
If label == rider, then check the head region.
Assign label "Helmet" or "No Helmet".
3️⃣ Number Plate Detection & OCR
If a rider is detected without a helmet, extract the number plate region.
Use EasyOCR for text recognition.
Preprocess the image (convert to grayscale, apply filters) before OCR.
🏗️ Model Training
Data Augmentation applied to improve performance.
Trained with optimized hyperparameters due to a limited dataset.
Saved and deployed the trained model.
🌐 Deployment
The model is deployed as a web app using Streamlit.
Supports both image and video input.
Video processing is done frame-by-frame for real-time detection.
🔧 Installation & Setup
in the repositary we have fine tuned yolo model and easy ocr model and also some testing videos and images and the streamlit code 
1️⃣ Clone the repository
git clone https://github.com/yourusername/helmet-numberplate-recognition.git
cd helmet-numberplate-recognition
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit web app
streamlit run app.py
