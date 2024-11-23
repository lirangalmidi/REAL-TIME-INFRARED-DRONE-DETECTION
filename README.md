# REAL-TIME INFRARED DRONE DETECTION
Final Project Documentation ,Electrical and Electronics Engineering , Tel Aviv University ,2025

## Algorithm Overview (YOLOv8)

YOLO (You Only Look Once) is a real-time, one-stage object detection algorithm known for its speed and accuracy. The algorithm divides the input image into a grid and simultaneously predicts bounding boxes and their associated class probabilities for each cell. 

Key features of YOLO include:
- **Speed:** YOLO processes images at high frame rates, making it suitable for real-time applications like drone detection.
- **End-to-End Training:** The algorithm is trained end-to-end, optimizing both the bounding box prediction and classification tasks in a single step.
- **Unified Framework:** YOLO’s single neural network design makes it efficient and compact compared to two-stage detectors like Faster R-CNN.

## Installation
Make sure you have Python 3.11 installed. You can download it from the official Python website.

### 1. Install Python 3.11

Verify the installation by entering the following command in the CMD:
```
python --version
```
You should see output similar to:
```
Python 3.11.x
```
### 2. Install Required Libraries

Install the core libraries for the project:
```
pip install numpy ultralytics
```
### 3. Install CUDA (Optional but **recommended**, for GPU Acceleration)
#### Why Use CUDA for GPU Acceleration?

Using CUDA with an NVIDIA GPU provides several key advantages:

**1.Faster Computations:** GPUs are optimized for parallel processing, making them much faster than CPUs for tasks such as training deep learning models or running inference in real time.
  
**2.Improved Performance for YOLO:** Running YOLOv8 on a GPU significantly improves its detection speed, allowing for real-time performance, which is crucial for applications like drone detection.

**3.Efficient Resource Utilization:** CUDA allows you to fully utilize the GPU's processing power, offloading compute-intensive operations from the CPU and improving overall system performance.

#### Requirements
**1. NVIDIA GPU:** Your system must have an NVIDIA GPU with a compute capability of at least 3.0.

**2. Driver Version:** Install the NVIDIA driver for your GPU . You can download the driver from the NVIDIA Drivers page. [for 11.8 version](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=Server2016)

### 4. Install PyTorch

Install PyTorch with GPU support (if CUDA is installed) or CPU-only version. Use [PyTorch installation guide](https://pytorch.org/)

![image](https://github.com/user-attachments/assets/96d4b73c-004d-4a0a-a6b3-37979ccad7b1)

Run the command in the picture in the CMD:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Creating the Project Directory
To organize the project effectively, follow these steps to create the necessary directory structure:
### 1. Open your file explorer and navigate to Drive C and create a new folder named yolo
![image](https://github.com/user-attachments/assets/a48424b9-7ef7-41a4-ad50-a4ae5efbfec6)
