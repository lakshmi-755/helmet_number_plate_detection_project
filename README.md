
# 🚀 Helmet and Number Plate Recognition for Motorcycles  

This project takes **images or videos** of motorcyclists as input and detects whether they are wearing a **helmet** or not.  
If a rider is found **without a helmet**, the system identifies their **number plate** and extracts the text for further action.  

---

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




