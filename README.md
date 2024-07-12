# Face Recognition Based Student Attendance and Monitoring System
Efficient attendance management is crucial for educational institutions, where traditional methods often prove inefficient, inaccurate, and prone to fraud. This project introduces a robust, face recognition-based attendance system designed to automate and enhance the process of student tracking and monitoring.

# Motivation
Traditional attendance-taking methods are time-consuming and occasionally unreliable. This project addresses these challenges by leveraging advanced facial recognition technology to streamline attendance processes, reduce administrative burdens, and provide real-time monitoring of student participation. By automating attendance, educators can focus more on teaching while ensuring accurate record-keeping and immediate alerts for irregularities such as students attending the wrong classes.

# Features
## Modules Overview
### Student Enrollment
Capture and store basic student information along with facial images for database enrollment.

### Face Detection and Recognition
Detect and recognize students' faces in real-time using computer vision algorithms for accurate attendance marking.

### Face Tracking
#### Introduction
We use the YOLOv8 object detection algorithm combined with tracking methods such as DeepOCSORT for real-time object tracking. This model processes videos, images, and live camera feeds.

#### Dataset
The MOT Challenge Dataset was used for training and evaluation, containing annotated video sequences.

#### Preprocessing
Data Augmentation: Horizontal flipping, scaling, cropping, and color jittering.
Normalization: Normalizing input images to pixel values between 0 and 1.
Annotation: Converting annotations to YOLO format.
#### YOLOv8
YOLOv8 is a CNN designed for real-time object detection, with a backbone network for feature extraction and a detection head for predicting bounding boxes and class probabilities. The SAHI library is used to improve performance on high-resolution images.

#### YOLO and DeepOCSORT
Integration of YOLOv8 and DeepOCSORT for detection and tracking:

#### Process video frames.
1. Detect objects using YOLOv8 and track them with DeepOCSORT.
2. Label detected objects with bounding boxes, IDs, and confidence scores.
3. Track detection times and determine if a student leaves the class.
4. Save results to a CSV file.
### Integration of Recognition and Tracking
#### Workflow
##### Video Upload:

Users upload a video for analysis.
##### Frame Analysis:
###### Initial Frame:
-Face Detection and Recognition:
-Detect and recognize faces using RetinaFace and FaceNet models.
-Associate recognized faces with objects detected in the frame.
-Object Detection and Tracking:
-Simultaneously detect and track objects using YOLOv8 and DeepOCSORT.
-Record object IDs, labels, and positions.
###### Subsequent Frames:
-Face Detection:
-Continuously detect faces in each frame.
-Face Recognition:
-Compare detected faces with previously recognized faces.
-Update tracking information for consistent individuals.
-start only if their is a change in the detected students.
#### Alert System:
Unrecognized Faces:
If a new face is detected that isn't recognized:
Capture the face image.
Alert the instructor for verification.

### Attendance Report
Generate comprehensive attendance reports that can be exported in formats like Excel and PDF, facilitated through a Power BI dashboard.

### Graphical User Interface (GUI)
Design a user-friendly web application to manage student details, receive alerts, and export attendance reports for each class section.

# Installation
Prerequisites
Python 3.x
Libraries: OpenCV, NumPy, Pandas, Flask, React. (List dependencies and installation commands)


# Setup
### Clone the repository:
<pre>
git clone https://github.com/your-username/attendance-system.git
cd attendance-system
</pre>
### Install dependencies:
<pre>
pip install -r requirements.txt
</pre>
### Run the application:
<pre>
python app.py
</pre>

Access the application at http://localhost:3000 in your web browser.

### Student Enrollment
![Enrollment](Screenshot%202024-05-04%20000429.png)
### Face Detection and Recognition
![Regocnition](Screenshot%202024-03-04%20160938.png)
### Attendance Tracking
![Tracking](Screenshot%202024-02-21%20002053.png)
### Attendance Reports
![Report](Screenshot%202024-04-16%20020818.png)

### Frame Result 
![Result](1.png)
### Demo
![Demo](D%20-%20Made%20with%20Clipchamp.mp4)

# Usage
1. Navigate to the web interface.
2. Register student details and assign classes.
3. Monitor real-time attendance and receive alerts for discrepancies.
4. Export attendance reports for record-keeping and analysis.
# Contributing
1. Fork the repository.
2. Create your feature branch (git checkout -b feature/NewFeature).
3. Commit your changes (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature/NewFeature).
5. Open a pull request.
