# 🍃 Tea Leaf Disease Classification
🔍 *Deep Learning • Computer Vision • Transfer Learning • Streamlit*

## 🚀 Tech Stack & Domains
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)
![Deep Learning](https://img.shields.io/badge/Domain-Deep%20Learning-brightgreen)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-blueviolet)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![Colab](https://img.shields.io/badge/Platform-Colab-yellow)

---

## 📘 Overview
This project classifies tea leaf images into three categories:
- **Brown Blight**
- **Algal Spot**
- **Healthy**

It demonstrates an **end-to-end deep learning workflow**, from image preprocessing to real-time prediction using a **Streamlit web interface**.

---

## 🎯 Problem Statement
Tea farmers face difficulty in identifying leaf diseases at an early stage. Manual inspection is time-consuming and often inaccurate, leading to:
- Crop yield loss  
- Increased treatment cost  
- Delayed disease control  

This project helps identify tea leaf diseases quickly using image-based deep learning classification.

---

## 💼 Business Use Cases
| Use Case | Description |
|--------|-------------|
| 🌱 Tea Plantation Management | Early detection reduces crop damage and improves yield |
| 🏭 Tea Business Operations | Prevent large-scale losses by detecting disease early |

---

## 🧠 Model Performance
| Model | Accuracy |
|------|----------|
| 🧠 VGG16 (Transfer Learning) | **96.6%** |

---

## 🗺️ Project Workflow

### 🧾 1 — Data Preprocessing
- Image resizing and normalization  
- Data augmentation  
- Train–Test split  

### 🧮 2 — Feature Engineering
- Feature extraction using pretrained CNN layers  
- Fine-tuning selected layers  

### 🤖 3 — Modeling
- CNN architecture  
- Transfer learning with **VGG16**  

### 📊 4 — Evaluation
- Accuracy and validation metrics  
- Model performance comparison  

### 🌐 5 — UI Development
- Streamlit app for real-time image upload  
- Instant disease prediction  

---


---

<summary>📸 Click to view Streamlit UI screenshots</summary>

#### Home Page  
![Home Page](https://github.com/user-attachments/assets/6f309ed6-8a00-4d95-8ab6-4757c22d933c)



#### Results Page 1 
![Result Page](https://github.com/user-attachments/assets/75a6dd13-5d87-4f59-a1df-e0e12e33847c)



#### Results Page 1 
![Result Page](https://github.com/user-attachments/assets/3e45d622-ba3c-4024-a1f7-c283a4de1054)


---


## 📁 Project Structure
```
Tea-Leaf-Disease-Classifier/  
│  
├── Test Dataset/  
│   └── Test Images 
│  
├── Trainig Code/  
│   └── Tea Leaf Deasease Training Codes.ipynb
│  
├── app/  
│   └── app.py  
│  
├── requirements.txt  
└── README.md  

```
---

## 🛠️ Installation & Execution

Clone repository:
```
git clone https://github.com/sarankumar74/Tea-Leaf-Disease-Classifier.git
cd Tea-Leaf-Disease-Classifier
```

Install dependencies:
```
pip install -r requirements.txt
```

Run Streamlit app:
```
streamlit run app/app.py
```
