# 🩺 Skin Disease Detection System

## 📌 Project Overview
This project is an **AI-based Skin Disease Detection System** using **Deep Learning**.  
It classifies different skin diseases from images using a **pretrained VGG16 model** with fine-tuning.

The system helps in **early detection of skin diseases** for medical diagnosis and research purposes.

---

## 🎯 Objective
- Detect skin diseases from images accurately  
- Use transfer learning with VGG16 to improve model performance  
- Compare training and fine-tuned accuracy  
- Save the model for future predictions

---

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Pretrained VGG16 (ImageNet weights)  

---

## 🧠 Model Architecture
The model is built using **Transfer Learning**:

1. **Data Preprocessing**
   - Images resized to 224x224 (VGG16 recommended)
   - Pixel values normalized
   - Data augmentation: Random flip, rotation, zoom, contrast  

2. **Base Model**
   - Pretrained **VGG16** without top layers
   - Initially all layers frozen except last 3 layers
   - Later fine-tuning last 20 layers

3. **Custom Layers**
   - GlobalAveragePooling2D
   - Dense layer with 256 neurons (ReLU)
   - Dropout (0.5) to prevent overfitting
   - Output Dense layer with **softmax** activation (number of skin disease classes)

---

## 📊 Training
- Optimizer: Adam  
- Loss function: Sparse Categorical Crossentropy  
- Metrics: Accuracy  
- Callbacks:
  - **EarlyStopping** to prevent overfitting  
  - **ReduceLROnPlateau** to adjust learning rate  

**Training Steps:**
1. Train only last layers of VGG16  
2. Fine-tune last 20 layers for better accuracy  

---

## 🗂️ Project Structure
