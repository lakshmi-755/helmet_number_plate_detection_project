
# 🚀 Helmet and Number Plate Recognition for Motorcycles  

This project takes **images or videos** of motorcyclists as input and detects whether they are wearing a **helmet** or not.  
If a rider is found **without a helmet**, the system identifies their **number plate** and extracts the text for further action.  

---
## 🚀 How it is working

### 🎬 Working Video
https://github.com/user-attachments/assets/23d5b4d6-c1db-4c3b-9657-68b27d515894


## 🚀 What the Project Does (Overall)

This project automatically monitors motorcyclists using images or videos to:

- Check whether a rider is **wearing a helmet**
- If a rider is **not wearing a helmet**, detect the **vehicle number plate**
- Extract the **number plate text** for further action (e.g., traffic enforcement)

---

## 🔄 How It Is Working (Step-by-Step)

### 1️⃣ Input (Image / Video)
The system accepts:
- 📷 An image, or
- 🎥 A video of road traffic  

If a video is provided, **frames are extracted** for processing.

---

### 2️⃣ Object Detection (YOLO Model)
A **YOLO-based deep learning model** is used to detect the following objects:

- 🏍️ Motorcycle  
- 👤 Rider  
- 🪖 Helmet  
- 🚫 No-Helmet  
- 🔢 Number Plate  

📌 YOLO scans each frame and draws **bounding boxes** around detected objects.

---

### 3️⃣ Helmet Detection Logic
The system performs the following checks:
- Is a **rider** detected?
- Is a **helmet** detected on the rider?

✔ If a helmet is present → **No action taken**  
❌ If a helmet is NOT present → **Proceed to number plate detection**

---

### 4️⃣ Number Plate Detection
When a **no-helmet rider** is identified:
- The system detects the **vehicle’s number plate**
- The **number plate region is cropped** from the image/frame

---

### 5️⃣ Text Extraction (OCR)
- The cropped number plate image is sent to an **OCR (Optical Character Recognition) engine**
- OCR converts the image into **readable text**

**Example output:**
- AP09 AB 1234
---

### 6️⃣ Output / Result
The system outputs:
- Bounding boxes around:
  - Rider
  - Helmet / No-Helmet
  - Number Plate
- Extracted **number plate text**

This data can be:
- 💾 Stored in a database
- 🚓 Used for issuing traffic challans
- 📤 Sent to authorities for further action



## 📌 Project Overview  
This project utilizes **Deep Learning** to detect motorcyclists and recognize whether they are wearing helmets.  
If a rider is not wearing a helmet, the system identifies their **number plate** and extracts the text.  

---

## 🎯 Objectives  
✔️ Detect motorcycles and riders in images/videos.  
✔️ Check whether the rider is wearing a helmet.  
✔️ Identify riders **without helmets** and extract their vehicle's number plate.  
✔️ Display the **recognized number plate text** for further processing.  

---

## 🛠️ Tech Stack  
- **Deep Learning Model:** YOLOv8n (fine-tuned for helmet detection)  
- **Libraries Used:**  
  - `Streamlit` - Web app deployment  
  - `OpenCV` - Image processing  
  - `NumPy` - Array operations  
  - `Pandas` - Data handling  
  - `Matplotlib` - Visualization  
  - `Imutils`, `PIL` - Image handling  
  - `EasyOCR` - Text recognition (for number plates)  

---

## 📂 Dataset  
A **custom dataset** was prepared by collecting publicly available images and annotating them using **Roboflow**.  
The dataset follows the YOLOv8 format:  
📁 **Training Set** - 70%  
📁 **Validation Set** - 20%  
📁 **Testing Set** - 10%  

---

## 🔍 Methodology  

### 1️⃣ Object Detection with YOLOv8n  
✔️ YOLOv8n is pre-trained on the **MS-COCO dataset**.  
✔️ Fine-tuned to detect **motorcycles, riders, and helmets**.  
✔️ **Custom labels:**  
   - 🟢 **Green** → Rider with helmet  
   - 🔴 **Red** → Rider without helmet  

### 2️⃣ Helmet Detection Logic  
✔️ If **label == rider**, then check the **head region**.  
✔️ Assign label **"Helmet"** or **"No Helmet"**.  

### 3️⃣ Number Plate Detection & OCR  
✔️ If a rider is detected **without a helmet**, extract the **number plate region**.  
✔️ Use **EasyOCR** for text recognition.  
✔️ Preprocess the image (convert to **grayscale, apply filters**) before OCR.  

---

## 🏗️ Model Training  
✔️ **Data Augmentation** applied to improve performance.  
✔️ Trained with **optimized hyperparameters** due to a limited dataset.  
✔️ Saved and deployed the trained model.  

---

## 🌐 Deployment  
✔️ The model is deployed as a **web app** using `Streamlit`.  
✔️ Supports both **image and video input**.  
✔️ **Video processing** is done frame-by-frame for real-time detection.  

---

## 🔧 Installation & Setup  
The repository includes:  
- **Fine-tuned YOLOv8 model**  
- **EasyOCR model**  
- **Testing videos and images**  
- **Streamlit web app code**  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/yourusername/helmet-numberplate-recognition.git
cd helmet-numberplate-recognition

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt

### 3️⃣ Run the Streamlit web app
```bash
streamlit run app.py




