# Forest-Fire-Prediction
 #### timeline
 jan 19 - started this repo  
 jan 20 - Found a dataset and started learning how to build a ML model  
 Feb 06 - cleaned Dataset  
 feb 27 - finished building model for classification using Random Forest   
 march 06 - finish building frontent using streamlit  
 march 07 - Implemented slider instead of inputbox
### Datasets used for each training process are given below
1.Forest_fire.ipynb - https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset
2.Satellite_detection.ipynb - https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset
3.camera_detection.ipynb - https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset
4.Fire_deetction.ipynb - https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11/dataset/2/download/yolov11

references-
1. https://github.com/ZephyrusBlaze/Wildfire-Detection
2. https://ieeexplore.ieee.org/document/10792919/
Due to storage isssues one model is uploaded in gdrive - https://drive.google.com/file/d/1A26azHn-fen7q1i-mjmpBhRI6Kb9fP_F/view?usp=sharing


# 🌲🔥 Forest Fire Prediction System

A machine learning and deep learning-based system designed to predict and detect forest fires using meteorological data, satellite images, and live video input. The project helps in early fire detection to assist forest departments and emergency services.

---

## 📖 About the Project

This project aims to implement a Forest Fire Prediction System using:

- **Machine Learning (ML)** for predicting fire risk based on weather data.
- **Convolutional Neural Networks (CNNs)** for detecting fire/smoke from satellite images.
- **YOLO (You Only Look Once)** for real-time fire detection in videos or live camera feeds.
- A **Flask web application** for an interactive frontend.

---

## ✨ Features

- 🔥 Predict fire probability using weather data.
- 🛰️ Detect fire/smoke from satellite images.
- 📹 Real-time fire detection from live webcam or uploaded videos.
- 📊 Visualizations: bounding boxes.
- 🖥️ User-friendly web interface.

---

## 🛠️ Tech Stack

| Component        | Tech Used                    |
|------------------|------------------------------|
| Programming      | Python                       |
| ML/DL Frameworks | Scikit-learn, PyTorch |
| Models           | Random Forest, ResNet-50, YOLOv11 |
| Web Framework    | Flask           |
| Visualization    | OpenCV, Matplotlib, Seaborn  |
| Dataset Sources  | Algerian Forest Fire Dataset, Custom Labeled Satellite Dataset |

---

## 📁 Project Structure
forest-fire-prediction/ │ 
├── datasets/ # Datasets 
├── model/ # Trained models (.h5 / .pt files) 
├── scripts/ # Data preprocessing and training scripts 
├── static/ # Static assets for Flask 
├── templates/ # HTML templates for Flask app 
├── app.py # Flask backend 
├── requirements.txt # Required Python packages 
└── README.md # Project overview

---

## 💻 Installation

1. **Clone the repository**
   ```bash
     git clone https://github.com/anugrahrk/Forest-fire.git
     cd forest-fire-prediction
2. **Install dependencies**
   ```bash
     pip install -r requirements.txt
3.**Download Trained Models**
-Place .h5 and .pt model files in the /model directory.

4.**Install dependencies**
   ```bash
      pip install -r requirements.txt

5.**Install dependencies**
  ```bash
     pip install -r requirements.txt

Go to http://localhost:5000

