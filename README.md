# REAL-TIME INFRARED DRONE DETECTION
Final Project Documentation ,Electrical and Electronics Engineering , Tel Aviv University ,2024-2025

## Algorithm Overview (YOLOv8)

YOLO (You Only Look Once) is a real-time, one-stage object detection algorithm known for its speed and accuracy. The algorithm divides the input image into a grid and simultaneously predicts bounding boxes and their associated class probabilities for each cell. 

Key features of YOLO include:
- **Speed:** YOLO processes images at high frame rates, making it suitable for real-time applications like drone detection.
- **End-to-End Training:** The algorithm is trained end-to-end, optimizing both the bounding box prediction and classification tasks in a single step.
- **Unified Framework:** YOLO’s single neural network design makes it efficient and compact compared to two-stage detectors like Faster R-CNN.

The YOLO (You Only Look Once) algorithm is a powerful tool for object detection, enabling real-time processing with high accuracy. It is widely used in various applications, such as real-time video processing for detecting and tracking moving objects, making it a key component in autonomous driving systems where pedestrians, vehicles, and traffic signs must be identified quickly and reliably. Additionally, YOLO is prevalent in security systems, including surveillance camera monitoring to detect intrusions or track suspicious individuals. Other applications include AI-based assistance systems, wildlife detection for environmental research, and smart traffic infrastructure management. Thanks to its efficiency, speed, and compatibility with diverse hardware, YOLO remains a leading solution in the field of artificial intelligence.

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
 
# Use the model to detect object
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

<img src="https://github.com/user-attachments/assets/45e5ea1c-4db0-4067-9cc5-d4d0811092a1" width="25%" height="auto">

### 3. Video Processing with YOLOv8

1. Download a Video into the directory where you want to perform the processing, for exmaple:



https://github.com/user-attachments/assets/9b79b3f3-937a-4637-9b9c-e6bb3275d14d



2. Open your text editor or IDE (e.g. Thony).

3. Create a new file and save it as video.py in the YOLOv8 folder.

Copy and paste the following code into the file:

``` python
# For video
from ultralytics import YOLO

# Load custom trained YOLOv8 model
model = YOLO("yolov8s.pt")

# Use the model to detect object and save the video to a folder
model.predict(source="dog.mp4", show=True, save=True, project="output_folder", name="detected_video")
```
Similar to the images, and by repeating the steps in a similar manner, we will obtain the analysis of the video:



https://github.com/user-attachments/assets/881c724a-a7fa-494a-a6da-81d9a6987449


### 4. Real-time Processing with YOLOv8

**Option 1- ip secuirty camera real time detection**

We wanted to test the algorithm in real-time, so we used  URL of an IP camera and use the python code: (you can replace with your url)

(we called it live.py and run it in the cmd in the same manner.)
``` python
from ultralytics import YOLO
import cv2

# Load custom trained YOLOv8 model
model = YOLO("yolov8s.pt")

# MJPEG stream URL (replace with the actual URL)
mjpg_url = "https://s123.ipcamlive.com/streams/7b5ahr6smisyvlu8d/stream.m3u8"

# MJPEG stream URL (replace with the actual URL)
#mjpg_url = "https://s34.ipcamlive.com/streams/22gv5xdbajoip2w1u/stream.m3u8"

# Open the MJPEG stream using OpenCV
cap = cv2.VideoCapture(mjpg_url)

# Check if the stream was opened successfully
if not cap.isOpened():
    print("Error: Unable to connect to the MJPEG stream.")
    exit()

# Create a resizable window for displaying the video
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

# Loop through the frames from the MJPEG stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Use the YOLO model to detect objects in the frame
    results = model(frame)

    # Extract bounding boxes and labels from the results
    for result in results[0].boxes.data:  # Assuming results[0] is the first frame result
        x1, y1, x2, y2 = result[:4]  # Coordinates of the bounding box
        label = result[5]  # Class label
        confidence = result[4]  # Confidence score

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Prepare the text with label and confidence score
        text = f"{model.names[int(label)]}: {confidence:.2f}"

        # Get the text size for the background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        # Set the background rectangle coordinates
        bg_x1, bg_y1 = int(x1), int(y1) - text_height - 10
        bg_x2, bg_y2 = int(x1) + text_width, int(y1)
        # Ensure coordinates are within the frame
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)

        # Draw a filled rectangle as the background for the text
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

        # Put the label and confidence score on top of the background rectangle
        cv2.putText(
            frame,
            text,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    # Display the frame with detected objects
    cv2.imshow("Detection", frame)

    # Resize the window (optional: you can set a specific width and height)
    cv2.resizeWindow("Detection", 1500, 600)  # Example dimensions

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

```


https://github.com/user-attachments/assets/c976aa65-048e-419e-a0c6-8bb13621975a

As mentioned from the algorithm's overview, one of its most common uses is in security systems. Later on, we will train the model to detect drones for our  project's purposes.
> Note: This method helped us evaluate the real-time performance of the algorithm's object detection capabilities, but we did not use this code for integration with the sensor.


**Option 2 -Webcam real time detection**

we also write code with input of a webcam (This script will help later to integrate with the sensor) for real time object detection:

``` python
# For webcam
from ultralytics import YOLO
 
# Load custom trained YOLOv8 model
model = YOLO("yolov8s.pt")
 
# Use the model to detect object
model.predict(source="1", show=True)
``` 

(we called it webcam.py and run it in the cmd in the same manner.)




## Training

Before we begin training the model, we will create a new folder named "yolo" (for the trained model), which will contain all the following scripts:

- image.py
- video.py
- webcam.py

In addition, for the algorithm to detect drones, we will need to train it on a labeled image dataset. Building a custom dataset can be a time-consuming process, often taking dozens or even hundreds of hours to collect images, label them, and export them in the proper format. Fortunately, Roboflow streamlines this process, making it as fast and straightforward as possible.
you can search for your dataset [here](https://universe.roboflow.com/browse/classic)
