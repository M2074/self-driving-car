# 🚗 Self-Driving Car Simulator

A complete end-to-end deep learning project that trains a convolutional neural network (CNN) to autonomously drive a car in a simulator using behavioral cloning.  
Inspired by NVIDIA’s groundbreaking research _End to End Learning for Self-Driving Cars_ (Bojarski et al., 2016).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Data Processing](#-data-processing)
- [Training Process](#-training-process)
- [Testing and Simulation](#-testing-and-simulation)
- [Results](#-results)
- [Demo](#-demo)
- [Dependencies](#-dependencies)
- [Future Improvements](#-future-improvements)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)

---

## 🎯 Overview

This project implements an autonomous driving system using deep learning techniques. The system:

- Collects driving data from a simulator
- Trains a CNN model using behavioral cloning
- Deploys the trained model to control a virtual car in real-time
- Uses computer vision and neural networks to predict steering angles

The approach follows the **end-to-end learning paradigm** demonstrated by NVIDIA, where raw pixels are directly mapped to steering commands, eliminating the need for separate lane detection, path planning, or rule-based control.

---

## ✨ Features

### 🧠 **Deep Learning Model**
- **NVIDIA-inspired CNN architecture** for end-to-end learning  
- **Behavioral cloning** approach to learn from human driving data  
- **Real-time inference** for autonomous driving  

### 📊 **Data Processing and Augmentation**
- **Image preprocessing** (cropping, color space conversion, normalization)  
- **Data augmentation** techniques:  
  - Random panning  
  - Zoom transformations  
  - Brightness adjustments  
  - Horizontal flipping  
- **Data balancing** to handle steering angle distribution  

### 🎮 **Real-time Simulation**
- **Socket.IO integration** for real-time communication with simulator  
- **Live telemetry processing** (speed, steering angle, camera feed)  
- **Autonomous control** with dynamic throttle adjustment  

### 📈 **Training and Monitoring**
- **Batch generation** for efficient training  
- **Training/validation split** for model evaluation  
- **Loss curve visualization** and monitoring  
- **Model checkpointing** and saving  

---

## 📁 Project Structure

```

selfDrivingSimulator/
├── 📄 main.py                  # Data reading example
├── 🏋️ TrainingSimulation.py    # Model training pipeline
├── 🎮 TestSimulation.py        # Real-time simulator interface
├── 🛠️ utils.py                 # Utility functions and model architecture
├── 📊 loss\_curve.png           # Training loss visualization
├── 🤖 model.h5                 # Trained model file
├── 🖼️ test.jpg                 # Test image
├── 📁 data/                    # Training data directory
│   ├── driving\_log.csv         # Driving data logs
│   └── IMG/                    # Training images

````

---

## 🚀 Installation

### Prerequisites
- Python 3.7+  
- TensorFlow/Keras  
- OpenCV  
- Udacity Self-Driving Car Simulator  

### Setup
```bash
# Clone the repository
git clone https://github.com/NareshP215/Self-Driving-Car-Simulation
cd selfDrivingSimulator

# Install dependencies
pip install -r requirements.txt
````

Or install manually:

```bash
pip install tensorflow opencv-python pandas numpy matplotlib scikit-learn imgaug flask-socketio eventlet pillow
```

---

## 🎯 Usage

### 1. **Data Collection**

* Launch the simulator in "Training Mode"
* Drive manually to collect training data
* Data gets saved as `driving_log.csv` with corresponding images in `IMG/`

### 2. **Model Training**

```bash
python TrainingSimulation.py
```

* Loads and balances training data
* Applies data augmentation
* Trains the CNN model
* Saves the trained model as `model.h5`
* Generates loss curve visualization

### 3. **Autonomous Driving**

```bash
python TestSimulation.py
```

* Loads the trained model
* Starts Socket.IO server on port 4567
* Connects to simulator in "Autonomous Mode"
* Processes camera feed and predicts steering angles
* Controls the car in real-time

### 4. **Simulator Setup**

1. Download Udacity Self-Driving Car Simulator
2. For training: Use "Training Mode" to collect data
3. For testing: Use "Autonomous Mode" and connect to `localhost:4567`

---

## 🏗️ Model Architecture

This project adapts the **NVIDIA CNN architecture** introduced in *End to End Learning for Self-Driving Cars*. The model directly maps raw camera images to steering commands using an end-to-end learning approach.

### NVIDIA-Inspired CNN

* **Input:** 66×200×3 YUV images (cropped, normalized, resized)
* **Convolutions:** 5 layers for feature extraction
* **Fully Connected:** 3 layers for control decision
* **Output:** Single steering angle value (regression)

```python
# Convolutional Layers
Conv2D(24, (5,5), strides=(2,2), activation='elu')
Conv2D(36, (5,5), strides=(2,2), activation='elu')
Conv2D(48, (5,5), strides=(2,2), activation='elu')
Conv2D(64, (3,3), strides=(1,1), activation='elu')
Conv2D(64, (3,3), strides=(1,1), activation='elu')

# Fully Connected Layers
Flatten()
Dense(100, activation='elu')
Dense(50, activation='elu')
Dense(10, activation='elu')
Dense(1, activation='linear')   # Steering angle output
```

📌 **Key Insight:**
The network automatically learns road features (lane boundaries, curves, etc.) directly from steering data, without explicit labels, making it robust to varied driving conditions.

---

## 🔄 Data Processing

### Image Preprocessing

1. Crop: remove sky/hood (60:135)
2. Convert RGB → YUV
3. Gaussian Blur (3×3 kernel)
4. Resize to 200×66 pixels
5. Normalize pixel values to \[0,1]

### Data Augmentation

* Random panning
* Zooming (1.0–1.2x)
* Brightness variation (0.4–1.2x)
* Horizontal flipping with steering inversion

### Data Balancing

* Analyze steering angle distribution
* Reduce bias from straight-driving data

---

## 🏋️ Training Process

**Configuration:**

* Batch Size: 64
* Epochs: 10
* Steps per Epoch: 300
* Validation Steps: 200
* Optimizer: Adam (lr=0.0001)
* Loss Function: Mean Squared Error (MSE)

**Split:**

* Training: 80% with augmentation
* Validation: 20% without augmentation

**Features:**

* Real-time batch generation
* On-the-fly augmentation
* Validation monitoring
* Loss curve visualization
* Model checkpointing

---

## 🎮 Testing and Simulation

* **Socket.IO** server for communication
* **Telemetry:** speed, steering, camera feed
* **Image preprocessing** pipeline
* **Model inference** for steering prediction
* **Dynamic throttle** control
* **Error handling** and safety

---

## 📊 Results

### Training Metrics

* Model successfully trains on simulator data
* Loss curves saved as `loss_curve.png`
* Validation loss monitoring prevents overfitting

### Performance

* Real-time inference (\~30 FPS)
* Smooth steering predictions
* Successful autonomous navigation
* Adaptive speed control

---

## 🎥 Demo Video

https://www.youtube.com/watch?v=Zr84TbaDZfM

```md
![Demo](demo.gif)
```

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt) for the full list.
Key libraries include:

* TensorFlow/Keras
* OpenCV
* Flask-SocketIO + Eventlet
* NumPy, Pandas, Matplotlib, Scikit-learn
* ImgAug, Pillow

---

## 🔮 Future Improvements

* Integrate lane detection for better road awareness
* Add traffic sign recognition
* Deploy on real-world RC car with Raspberry Pi
* Implement reinforcement learning for adaptive driving

---

## 🔧 Troubleshooting

* **Model not loading:** Check file paths in `TestSimulation.py`
* **Simulator connection issues:** Ensure port 4567 is available
* **Poor driving performance:** Collect more diverse training data
* **Training errors:** Verify data paths and image integrity

---

## 📚 References

- NVIDIA, 2016. *End to End Learning for Self-Driving Cars*  
[Read the paper (PDF)](end-to-end-dl-using-px.pdf)
