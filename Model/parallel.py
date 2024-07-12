import os
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from Model.recognize_face import recognize_face_with_database
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import threading

def send_email_with_photo(image_path, recognized_id):
    # Email configurations
    sender_email = 'espaya580@gmail.com'
    receiver_email = 'ayamahmoudk677@gmail.com'
    password = 'pxqz qwyj bkfc uypj'

    # Create message container
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = 'Unknown Student Detected'

    # Attach image
    with open(image_path, 'rb') as fp:
        img = MIMEImage(fp.read())
    img.add_header('Content-Disposition', 'attachment', filename=f'{recognized_id}_photo.jpg')
    msg.attach(img)

    # Send email
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:  # Update SMTP server address
        smtp.starttls()
        smtp.login(sender_email, password)
        smtp.send_message(msg)
        smtp.quit()


def recognition_worker(face, db_path):
    return recognize_face_with_database(face, db_path)

def calculate_iou(box1, box2):
    # Calculate coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

def match_bounding_boxes(bboxes1, bboxes2, threshold):
    matched_pairs = []
    
    for i, box1 in enumerate(bboxes1):
        best_match = None
        best_iou = 0
        
        for j, box2 in enumerate(bboxes2):
            iou = calculate_iou(box1, box2)
            if iou > best_iou:
                best_iou = iou
                best_match = j
        
        if best_iou >= threshold:
            matched_pairs.append((i, best_match))
    
    return matched_pairs

def match_recognized_faces_with_tracked_persons(recognized_faces, tracked_persons, threshold):
    matched_pairs = []

    for person_id, person_data in tracked_persons.items():
        best_match = None
        best_iou = 0

        for recognized_face in recognized_faces:
            iou = calculate_iou(person_data["xyxy"], recognized_face["xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_match = recognized_face

        if best_iou >= threshold:
            matched_pairs.append((person_id, best_match))

    return matched_pairs


def process_frame(frame, detection_model, tracker, tracked_faces, detection_results, db_path, csv_file):
    # Previous frame's detection results count
    prev_detection_count = len(tracked_faces)

    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    num_predictions = len(result.object_prediction_list)
    dets = np.zeros([num_predictions, 6], dtype=np.float32)
    for ind, object_prediction in enumerate(result.object_prediction_list):
        dets[ind, :4] = np.array(object_prediction.bbox.to_xyxy(), dtype=np.float32)
        dets[ind, 4] = object_prediction.score.value
        dets[ind, 5] = object_prediction.category.id

    tracks = tracker.update(dets, frame)

    if tracks.shape[0] != 0:
        xyxys = tracks[:, 0:4].astype('int')  # float64 to int
        ids = tracks[:, 4].astype('int')  # float64 to int
        confs = tracks[:, 5].round(decimals=2)
        clss = tracks[:, 6].astype('int')  # float64 to int

        # Prepare threads for recognition
        recognition_threads = []
        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
            face = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            t = threading.Thread(target=recognition_worker, args=(face, db_path))
            recognition_threads.append((t, id))
            t.start()

        # Join threads and update tracked_faces and detection_results
        for t, id in recognition_threads:
            t.join()
            recognized_id, first_name, last_name, academic_year, department = t.result()
            tracked_faces[id]["Recognized ID"] = recognized_id

            if recognized_id == "unknown":
                # Send alert email
                cv2.imwrite('unknown_person.jpg', face)  # Save the unknown face
                send_email_with_photo('unknown_person.jpg', recognized_id)

            new_detection_result = pd.DataFrame({
                "ID": [id],
                "Recognized ID": [recognized_id],
                "First name": [first_name],
                "Last name": [last_name],
                "Academic_year": [academic_year],
                "Department": [department],
                "First Detection Time": [tracked_faces[id]["First Detection Time"]],
                "Last Detection Time": [datetime.now()],
                "Confidence": [conf],
                "Left Class": [False]
            })
            detection_results = pd.concat([detection_results, new_detection_result], ignore_index=True)

            # Update the frame with bounding box and recognized ID
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {id}, Recognized ID: {recognized_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    detection_results.to_csv(csv_file, index=False)
    return tracked_faces, detection_results
