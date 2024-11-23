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
### 1. Open your file explorer and navigate to Drive C and create a new folder named YOLOv8

![image](https://github.com/user-attachments/assets/e40a5b5a-2cfa-4bc6-af14-bf2e8773e9a2)


### 2. Image Processing with YOLOv8
Once the project folder is set up, you can begin processing images using the YOLOv8 algorithm.

Before training, the YOLOv8 algorithm comes pre-trained on the COCO dataset (if you use the default weights, such as yolov8n.pt, yolov8s.pt, etc.). This means it can already detect 80 common object categories, including:

- Vehicles: cars, trucks, buses, bicycles, motorcycles, airplanes, boats.

- People: humans (general person category).

- Animals: dogs, cats, birds, horses, cows, elephants, etc.

- Objects: backpacks, umbrellas, cell phones, laptops, cups, bottles, etc.
  
**Create a Python File for Detection**

Inside the yolo directory, create a Python file to run the detection process. Follow these steps:

1. Download an image into the directory where you want to perform the processing, for exmaple:
<img src="https://github.com/user-attachments/assets/8ea75dfc-d691-4ddd-b8a3-bc49e7267dea" width=25% height=25%>


2. Open your text editor or IDE (e.g. Thony).

3. Create a new file and save it as image.py in the YOLOv8 folder.

Copy and paste the following code into the file:

``` python
from ultralytics import YOLO
 
# Load our custom drone model
model = YOLO('yolov8s.pt')
 
# Use the model to detect object - drone
model.predict(source="dog.jpg", save=True, show=True)
```
Now your directory should look like this:

![image](https://github.com/user-attachments/assets/3b995230-9940-44a1-86ab-237ff1fd61fa)

Write `cmd` in the folder path.

![image](https://github.com/user-attachments/assets/c6ce2ffe-a429-425d-a0fd-b3602ddedbf5)

In the cmd window, enter the command: `python image.py`

![image](https://github.com/user-attachments/assets/4d30e2ea-fc9e-4ea4-9c93-f6f7f538a714)

Press Enter and wait for the execution to finish. Finally, the directory should look like this:

![image](https://github.com/user-attachments/assets/b0069371-a964-448e-8760-d172a54c4023)

Go to `run -> detect -> predict` and there you will find the image after processing and object detection:

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/68c2ec13-0f0e-41c1-9f92-736e233a27d8" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/45e5ea1c-4db0-4067-9cc5-d4d0811092a1" width="300"></td>
  </tr>
</table>




